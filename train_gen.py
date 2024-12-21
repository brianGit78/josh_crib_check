import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from file_sync import FileManager
import creds

#file operations
file_manager = FileManager(creds.model_name)
file_manager.create_local_directories()
file_manager.sync_source(creds.nas_user, creds.nas_password, creds.nas_host, creds.nas_path) #remove this if you already copied your files to this directly in the specified structure
file_manager.split_data_for_validation(os.path.join(file_manager.local_path_training_data, "true"), os.path.join(file_manager.local_path_validation_data, "true"))
file_manager.split_data_for_validation(os.path.join(file_manager.local_path_training_data, "false"), os.path.join(file_manager.local_path_validation_data, "false"))

def create_model(input_shape=(256, 256, 1)):
    return Sequential([
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

# Load and preprocess the mask
mask = load_img('crib_mask.png', color_mode='grayscale', target_size=(256, 256))
mask = img_to_array(mask) / 255.0  # Convert to [0, 1] range

def preprocess_input(img):
    img = img / 255.0
    img = img * mask
    return img

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.9, 1.1],  # much closer to normal lighting
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

target_size = (256, 256)
batch_size = 64

train_generator = train_datagen.flow_from_directory(
    file_manager.local_path_training_data,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'   
)
print(train_generator.class_indices)

validation_generator = validation_datagen.flow_from_directory(
    file_manager.local_path_validation_data,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale' 
)

model = create_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),  #monitor='val_loss'
    ModelCheckpoint('best_model.keras', save_best_only=True)]

model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=callbacks
)

evaluation = model.evaluate(validation_generator)
print(f"Validation Loss: {evaluation[0]:.4f}, Validation Accuracy: {evaluation[1]:.4f}")

#remove the model file if it exists
file_manager.archive_model_file()

# Load the best weights saved by ModelCheckpoint
model.load_weights('best_model.keras')
# Then save the model to your desired path
model.save(file_manager.model_file_path)
