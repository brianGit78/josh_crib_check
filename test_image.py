import numpy as np
import cv2, os
from tensorflow.keras.models import load_model
from file_sync import FileManager
import creds

file_manager = FileManager(creds.model_name)
model = load_model(file_manager.model_file_path)

def preprocess_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Resize to match training size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256))
    img = img.astype('float32') / 255.0 # Convert to float and normalize (0-1)
    img = np.expand_dims(img, axis=0) # Expand dimensions to form a batch of one image
    return img

# Path to the test image
test_image_path = os.path.join(file_manager.local_path_validation_data, "false", "JoshNanit.20241212_185800293.jpg")
#test_image_path = 'validation_data/true/JoshNanit.20241111_013000303.jpg'

# Preprocess the test image
test_img = preprocess_image(test_image_path)

# Get prediction
prediction = model.predict(test_img)[0][0]

# Determine class based on threshold
threshold = 0.5
if prediction > threshold:
    print(f"Prediction: {prediction:.4f} - The model thinks: TRUE (in the crib)")
else:
    print(f"Prediction: {prediction:.4f} - The model thinks: FALSE (not in the crib)")
