import os, cv2, time, datetime, asyncio, logging
from logging.handlers import RotatingFileHandler
import numpy as np
import torch
from PIL import Image
from toggle_josh_crib import JoshAlertAsync
from file_sync import FileManager
from pt_cnn import SimpleCNN
import creds

file_manager = FileManager(creds.model_name)
# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler('logs/vid_proc_svc.log', maxBytes=5*1024*1024, backupCount=5)  # 5 MB per file, keep 5 backups
console_handler = logging.StreamHandler()

# Create formatters and add them to handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Initialize model
#device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
model.load_state_dict(torch.load(file_manager.model_file_path, map_location=device))
model.eval()

# Load mask once
mask_pil = Image.open('crib_mask.png').convert('L').resize((256, 256))
mask_array = np.array(mask_pil) / 255.0
mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).float()  # (1,1,256,256)
mask_tensor = mask_tensor.to(device)

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (256, 256))
    frame = cv2.equalizeHist(frame)  # Apply histogram equalization
    frame = frame.astype("float32") / 255.0
    #frame = np.expand_dims(frame, axis=-1)  # (256,256,1)
    #frame = np.expand_dims(frame, axis=0)   # (1,256,256,1)
    #return frame
    frame = torch.from_numpy(frame).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims (1,1,256,256)
    # Apply the mask
    frame = frame * mask_tensor
    return frame.to(device)

def connect_stream(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        logging.error("Failed to connect to the stream. Retrying in 5 seconds...")
        time.sleep(5)
        return connect_stream(url)
    return cap

async def cv_proc():
    josh_alert = JoshAlertAsync(home_assistant_url = creds.home_assistant_url, 
                       ha_access_token = creds.ha_access_token, 
                       ha_entity_id = creds.ha_entity_id,
                       update_interval=120
                       )
    
    # Start periodic state checker
    await josh_alert.start_periodic_check()

    # Connect to RTSP stream
    cap = connect_stream(creds.rtsp_url)

    threshold = 0.5
    check_interval = 3
    last_check_time = time.time()
    in_crib_count = 0
    not_in_crib_count = 0

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
                #prediction = model.predict(processed_frame, verbose=0)[0][0]
                with torch.no_grad():
                    prediction = torch.sigmoid(model(processed_frame)).item()

                if prediction > threshold:
                    in_crib_count += 1
                    not_in_crib_count = 0
                    # Turn on if we've reached threshold
                    if in_crib_count >= 3:
                        await josh_alert.turn_on_helper()
                        logging.info(f"Josh is IN the crib - Prediction: {prediction:.4f}")
                else:
                    not_in_crib_count += 1
                    in_crib_count = 0
                    # Turn off if we've reached threshold
                    if not_in_crib_count >= 3:
                        await josh_alert.turn_off_helper()
                        logging.info(f"Josh is NOT in the crib - Prediction: {prediction:.4f}")

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
