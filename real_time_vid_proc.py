import cv2, time
import numpy as np
from tensorflow.keras.models import load_model
from toggle_josh_crib import JoshAlert
import creds

# Load the trained model
model = load_model('/home/brian/josh_crib_check/crib_model.keras')

# Initialize JoshAlert class
josh_alert = JoshAlert()

def preprocess_frame(frame):
    # Resize the frame
    frame = cv2.resize(frame, (256, 256))
    # Normalize pixel values
    frame = frame / 255.0
    # Expand dimensions to match model input (batch of 1)
    return np.expand_dims(frame, axis=0)

def connect_stream(url):
    """Attempt to connect to the video stream and return the capture object."""
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Failed to connect to the stream. Retrying in 5 seconds...")
        time.sleep(5)
        return connect_stream(url)
    return cap

def main():
    cap = connect_stream(creds.rtsp_url)
    check_interval = 3
    threshold = 0.5  # Prediction threshold for deciding if Josh is in the crib

    while True:
        try:
            # Attempt to read a frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame. Reconnecting...")
                cap.release()
                cap = connect_stream(creds.rtsp_url)
                continue

            # Preprocess frame and predict
            processed_frame = preprocess_frame(frame)
            prediction = model.predict(processed_frame, verbose=0)[0][0]

            # Toggle alert based on prediction
            if prediction > threshold:
                josh_alert.turn_on_helper()
                print(f"Josh is IN the crib - Prediction: {prediction:.4f}")
            else:
                josh_alert.turn_off_helper()
                print(f"Josh is NOT in the crib - Prediction: {prediction:.4f}")

            # Wait before next check
            time.sleep(check_interval)

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
            cap.release()
            cap = connect_stream(creds.rtsp_url)

    cap.release()

if __name__ == '__main__':
    main()
