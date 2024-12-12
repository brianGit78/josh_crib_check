import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('/home/brian/josh_crib_check/crib_model.keras')

def preprocess_image(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image at {image_path}")

    # Resize to match training size
    img = cv2.resize(img, (256, 256))
    
    # Convert to float and normalize (0-1)
    img = img.astype('float32') / 255.0
    
    # Expand dimensions to form a batch of one image
    img = np.expand_dims(img, axis=0)
    return img

# Path to the test image
test_image_path = 'validation_data/true/JoshNanit.20241111_013000303.jpg'

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
