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
    
    # Normalize pixel values if needed (e.g., divide by 255)
    frame = frame / 255.0
    
    # Expand dimensions to match the model's input
    frame = np.expand_dims(frame, axis=0)
    
    return frame


def main():
    # Initialize video stream
    cap = cv2.VideoCapture(creds.rtsp_url)

    # Set the interval for checking the stream (in seconds)
    check_interval = 3

    while True:
        try:
            if not cap.isOpened():
                print("Reconnecting to the stream...")
                cap.release()
                cv2.destroyAllWindows()
                cap = cv2.VideoCapture(creds.rtsp_url)
                time.sleep(5)  # Wait before retrying

            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame. Retrying...")
                cap.release()
                cv2.destroyAllWindows()
                time.sleep(5)  # Wait before retrying
                cap = cv2.VideoCapture(creds.rtsp_url)
                continue

            # Preprocess the frame as done during training
            processed_frame = preprocess_frame(frame)  # Implement this function based on your training preprocessing
            
            # Predict
            prediction = model.predict(processed_frame)
            if prediction[0] > 0.5:  # Assuming 1 is 'true' for son in crib
                josh_alert.turn_on_helper()  # Son is in the crib
                print(f"Josh is IN the crib - Prediction: {prediction[0]}")
            else:
                josh_alert.turn_off_helper()  # Son is not in the crib
                print(f"Josh is NOT in the crib - Prediction: {prediction[0]}")
            
            # Wait for the next interval
            time.sleep(check_interval)

        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5)  # Wait before retrying

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
