import cv2, time, datetime
import numpy as np
from tensorflow.keras.models import load_model
from toggle_josh_crib import JoshAlert
from file_sync import FileManager
import creds_care_chair as creds

file_manager = FileManager(creds.model_name)
model = load_model(file_manager.model_file_path)
josh_alert = JoshAlert()

def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (256, 256))
    frame = frame.astype("float32") / 255.0
    frame = np.expand_dims(frame, axis=-1)  # (256,256,1)
    frame = np.expand_dims(frame, axis=0)   # (1,256,256,1)
    return frame

def connect_stream(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Failed to connect to the stream. Retrying in 5 seconds...")
        time.sleep(5)
        return connect_stream(url)
    return cap

def main():
    cap = connect_stream(creds.rtsp_url)
    threshold = 0.5
    check_interval = 3
    last_check_time = time.time()
    not_in_crib_count = 0

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print(f"{datetime.datetime.now()} - Failed to read frame. Reconnecting...")
                cap.release()
                cap = connect_stream(creds.rtsp_url)
                continue

            # Only process every check_interval seconds
            current_time = time.time()
            if current_time - last_check_time >= check_interval:
                last_check_time = current_time

                processed_frame = preprocess_frame(frame)
                prediction = model.predict(processed_frame, verbose=0)[0][0]

                if prediction > threshold:
                    not_in_crib_count = 0
                    josh_alert.turn_on_helper()
                    print(f"{datetime.datetime.now()} - Care Chair Is Occupied - Prediction: {prediction:.4f}")
                else:
                    not_in_crib_count += 1
                    if not_in_crib_count >= 3:
                        josh_alert.turn_off_helper()
                        print(f"{datetime.datetime.now()} - Care Chair is not Occupied - Prediction: {prediction:.4f}")

        except Exception as e:
            print(f"{datetime.datetime.now()} - An error occurred: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            cap.release()
            cap = connect_stream(creds.rtsp_url)

    cap.release()

if __name__ == '__main__':
    main()