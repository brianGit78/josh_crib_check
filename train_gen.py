import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (
    Conv2D, 
    MaxPooling2D, 
    Flatten, 
    Dense, 
    Input)
from tensorflow.keras.callbacks import EarlyStopping

from file_sync import FileManager
import creds

# Enable mixed precision
#tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Configure TensorFlow for parallel processing
#tf.config.threading.set_inter_op_parallelism_threads(8)
#tf.config.threading.set_intra_op_parallelism_threads(8)

def create_model(input_shape=(256, 256, 3)):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    return model

# Sync the training data from the remote server to the local machine
file_manager = FileManager(creds.nas_user,
                    creds.nas_password,
                    creds.nas_conn, 
                    creds.nas_path_td_src, 
                    creds.local_path)

file_manager.create_local_directories()
file_manager.start_sync_source()
file_manager.remove_validation_files_from_training_data()
file_manager.remove_model_file()

# Optional: Set TensorFlow to not use GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Define the data generator for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,  # Increased rotation range
    width_shift_range=0.3,  # Increased width shift range
    height_shift_range=0.3,  # Increased height shift range
    shear_range=0.3,  # Increased shear range
    zoom_range=0.3,  # Increased zoom range
    horizontal_flip=True,
    vertical_flip=True,  # Added vertical flip
    fill_mode='nearest'
)

# Data generator for validation
validation_datagen = ImageDataGenerator(rescale=1./255)

# Resize images to a smaller size
target_size = (256, 256)

train_generator = train_datagen.flow_from_directory(
    file_manager.local_path_training_data,
    target_size=target_size,
    batch_size=64,
    class_mode='binary'
)

validation_generator = validation_datagen.flow_from_directory(
    file_manager.local_path_validation_data,
    target_size=target_size,
    batch_size=64,
    class_mode='binary'
)

# Create model instance
model = create_model()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping callback
# this prevents the model from overfitting by monitoring the validation loss if it does not improve in 5 passes
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Evaluate the model
evaluation = model.evaluate(validation_generator)
print(f"Validation Loss: {evaluation[0]}, Validation Accuracy: {evaluation[1]}")

model.save(file_manager.model_file_path)  # Save the trained model in the native Keras format
