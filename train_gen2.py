import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from file_sync import FileManager
import creds

# Keep FileManager functionality as requested
file_manager = FileManager(
    creds.nas_user,
    creds.nas_password,
    creds.nas_conn,
    creds.nas_path_td_src,
    creds.local_path
)

# Prepare local directories and sync data
file_manager.create_local_directories()
file_manager.start_sync_source()
file_manager.remove_validation_files_from_training_data()
file_manager.remove_model_file()

# Optional: Disable GPU if desired
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_model(input_shape=(256, 256, 3)):
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
    rotation_range=2,          # very small rotation
    width_shift_range=0.05,    # slight horizontal shift
    height_shift_range=0.05,   # slight vertical shift
    shear_range=0.05,          # small shear
    zoom_range=0.1,            # slight zoom
    horizontal_flip=True,      # horizontal flip might still be okay
    brightness_range=(0.9, 1.1) # small brightness adjustments
)

validation_datagen = ImageDataGenerator(rescale=1./255)

target_size = (256, 256)
batch_size = 64

train_generator = train_datagen.flow_from_directory(
    file_manager.local_path_training_data,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'
)
print(train_generator.class_indices)

validation_generator = validation_datagen.flow_from_directory(
    file_manager.local_path_validation_data,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary'
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

model.save(file_manager.model_file_path)
