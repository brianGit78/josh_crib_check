import os, logging
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from file_sync import FileManager
import creds

# Configure logging
if not os.path.exists('logs'):
    os.makedirs('logs')

logging.basicConfig(
    filename='logs/train_gen.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
   )

#file operations
logging.info('Initializing FileManager')
file_manager = FileManager(creds.model_name)
file_manager.create_local_directories()
logging.info('Syncing source files')
file_manager.sync_source(creds.nas_user, creds.nas_password, creds.nas_host, creds.nas_path) #remove this if you already copied your files to this directly in the specified structure
logging.info('Splitting data for validation')
file_manager.split_data_for_validation(os.path.join(file_manager.local_path_training_data, "true"), os.path.join(file_manager.local_path_validation_data, "true"))
file_manager.split_data_for_validation(os.path.join(file_manager.local_path_training_data, "false"), os.path.join(file_manager.local_path_validation_data, "false"))

def create_model(input_shape=(256, 256, 1)):
    logging.info('Creating model with input shape %s', input_shape)
    model = Sequential([
        Input(shape=input_shape), # Input layer
        Conv2D(32, (3, 3), activation='relu'), # Convolutional layer
        MaxPooling2D(),     # Pooling layer
        Conv2D(64, (3, 3), activation='relu'), # Convolutional layer
        MaxPooling2D(),    # Pooling layer
        Conv2D(128, (3, 3), activation='relu'), # Convolutional layer
        MaxPooling2D(),   # Pooling layer
        Flatten(), # Flatten the output of the convolutional layers
        Dense(128, activation='relu'), # Dense layer
        Dense(1, activation='sigmoid') # Output layer
    ])
    logging.info('Model created successfully')
    return model

# Load and preprocess the mask
logging.info('Loading and preprocessing mask')
mask = load_img('crib_mask.png', color_mode='grayscale', target_size=(256, 256)) # Load the mask
mask = img_to_array(mask) / 255.0  # Convert to numpy array and normalize
logging.info('Mask loaded and preprocessed successfully')

def preprocess_input(img):
    logging.info('Preprocessing input image')
    img = img / 255.0 # Normalize the image
    img = img * mask # Apply the mask
    logging.info('Input image preprocessed successfully')
    return img

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # Apply the mask to the images
    rotation_range=2, # Rotate the image up to 2 degrees
    width_shift_range=0.05, # Shift the image up to 5% of its width
    height_shift_range=0.05, # Shift the image up to 5% of its height
    shear_range=0.05,   # Shear the image
    zoom_range=0.1, # Zoom in slightly
    horizontal_flip=True, # Flip horizontally
    brightness_range=[0.9, 1.1],  # much closer to normal lighting
    fill_mode='nearest' # Fill in missing pixels with the nearest filled pixel
)

validation_datagen = ImageDataGenerator(rescale=1./255) # No augmentation for validation data
target_size = (256, 256) # All images will be resized to 256x256
batch_size = 64 # Number of images to process at a time

train_generator = train_datagen.flow_from_directory(
    file_manager.local_path_training_data,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary', # Binary classification
    color_mode='grayscale' 
) # Set as training data

validation_generator = validation_datagen.flow_from_directory(
    file_manager.local_path_validation_data,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary', # Binary classification
    color_mode='grayscale' 
) # Set as validation data

model = create_model() # Create the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # Compile the model

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),  # Stop early if the validation loss stops improving
    ModelCheckpoint('best_model.keras', save_best_only=True)] # Save the best model during training

model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=callbacks
)

evaluation = model.evaluate(validation_generator) # Evaluate the model on the validation data
print(f"Validation Loss: {evaluation[0]:.4f}, Validation Accuracy: {evaluation[1]:.4f}")

#remove the model file if it exists
file_manager.archive_model_file()

# Load the best weights saved by ModelCheckpoint
model.load_weights('best_model.keras')
# Then save the model to your desired path
model.save(file_manager.model_file_path)
