import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from file_sync import FileManager
import creds_care_chair as creds

#file operations
file_manager = FileManager(creds.model_name)
file_manager.create_local_directories()
file_manager.sync_source(creds.nas_user, creds.nas_password, creds.nas_host, creds.nas_path)
file_manager.split_data_for_validation(os.path.join(file_manager.local_path_training_data, "true"), os.path.join(file_manager.local_path_validation_data, "true"))
file_manager.split_data_for_validation(os.path.join(file_manager.local_path_training_data, "false"), os.path.join(file_manager.local_path_validation_data, "false"))

# Optional: Disable GPU if desired
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_model(input_shape=(256, 256, 1)):  # Changed to 1 channel
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

# Subtle data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=(0.9, 1.1)
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

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

evaluation = model.evaluate(validation_generator)
print(f"Validation Loss: {evaluation[0]:.4f}, Validation Accuracy: {evaluation[1]:.4f}")

#remove the model file if it exists
file_manager.archive_model_file()
model.save(file_manager.model_file_path)
