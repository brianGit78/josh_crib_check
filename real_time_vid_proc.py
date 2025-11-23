import os, cv2, time, datetime, asyncio, logging, json
from collections import deque
from logging.handlers import RotatingFileHandler

import torch
from PIL import Image

import creds
from device_utils import describe_device, select_device
from file_sync import FileManager
from preprocessing import build_transforms
from pt_cnn import CribMobileNet
from toggle_josh_crib import JoshAlertAsync

file_manager = FileManager(creds.model_name)
# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# 5 MB per file, keep 5 backups
file_handler = RotatingFileHandler('logs/vid_proc_svc.log', maxBytes=5 * 1024 * 1024, backupCount=5)
console_handler = logging.StreamHandler()

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Initialize model and pick the strongest available device (RTX 5090 > 4070 > others)
device = select_device()
if device.type == 'cuda':
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

logger.info("Using device: %s", describe_device(device))

model = CribMobileNet(pretrained=False).to(device)
model.load_state_dict(torch.load(file_manager.model_file_path, map_location=device))
model.eval()

preprocess = build_transforms('crib_mask.png', train=False)


def load_thresholds(default_on=0.8, default_off=0.6):
    threshold_on, threshold_off = default_on, default_off
    if os.path.exists(file_manager.thresholds_file_path):
        try:
            with open(file_manager.thresholds_file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
                threshold_on = float(payload.get("threshold_on", threshold_on))
                threshold_off = float(payload.get("threshold_off", threshold_off))
                logging.info(
                    "Loaded calibrated thresholds: on=%.3f off=%.3f (base=%.3f)",
                    threshold_on,
                    threshold_off,
                    payload.get("base_threshold"),
                )
        except Exception as exc:
            logging.warning("Failed to load calibrated thresholds, using defaults: %s", exc)

    return threshold_on, threshold_off


def preprocess_frame(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    tensor = preprocess(pil_image).unsqueeze(0)
    return tensor.to(device, non_blocking=device.type == 'cuda')

def connect_stream(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logging.error("Failed to connect to the stream. Retrying in 5 seconds...")
        time.sleep(5)
        return connect_stream(url)
    return cap

async def cv_proc():
    josh_alert = JoshAlertAsync(
        home_assistant_url=creds.home_assistant_url,
        ha_access_token=creds.ha_access_token,
        ha_entity_id=creds.ha_entity_id,
        update_interval=120,
    )
    
    # Start periodic state checker
    await josh_alert.start_periodic_check()

    # Connect to RTSP stream
    cap = connect_stream(creds.rtsp_url)

    threshold_on, threshold_off = load_thresholds(default_on=0.8, default_off=0.6)
    check_interval = 3
    last_check_time = time.time()
    in_crib_count = 0
    not_in_crib_count = 0
    prediction_history = deque(maxlen=5)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error(f"{datetime.datetime.now()} - Failed to read frame. Reconnecting...")
                cap.release()
                cap = connect_stream(creds.rtsp_url)
                continue

            current_time = time.time()
            if current_time - last_check_time >= check_interval:
                last_check_time = current_time

                processed_frame = preprocess_frame(frame)
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type, enabled=device.type == 'cuda'
                ):
                    prediction = torch.sigmoid(model(processed_frame)).item()

                prediction_history.append(prediction)
                smoothed_prediction = sum(prediction_history) / len(prediction_history)

                if smoothed_prediction >= threshold_on:
                    in_crib_count += 1
                    not_in_crib_count = 0
                    if in_crib_count >= 3:
                        await josh_alert.turn_on_helper()
                        logging.info(
                            "Josh is IN the crib - raw: %.4f | smoothed: %.4f",
                            prediction,
                            smoothed_prediction,
                        )
                elif smoothed_prediction <= threshold_off:
                    not_in_crib_count += 1
                    in_crib_count = 0
                    if not_in_crib_count >= 3:
                        await josh_alert.turn_off_helper()
                        logging.info(
                            "Josh is NOT in the crib - raw: %.4f | smoothed: %.4f",
                            prediction,
                            smoothed_prediction,
                        )
                else:
                    in_crib_count = 0
                    not_in_crib_count = 0

    except KeyboardInterrupt:
        print("Shutting down stream...")
    finally:
        # Cleanup
        cap.release()
        await josh_alert.stop_periodic_check()

def main():
    asyncio.run(cv_proc())    

if __name__ == '__main__':
    main()
