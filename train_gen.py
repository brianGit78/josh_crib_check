import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from file_sync import FileManager
import creds

#file operations
file_manager = FileManager(creds.model_name)
file_manager.create_local_directories()
file_manager.sync_source(creds.nas_user, creds.nas_password, creds.nas_host, creds.nas_path)
file_manager.split_data_for_validation(os.path.join(file_manager.local_path_training_data, "true"), os.path.join(file_manager.local_path_validation_data, "true"))
file_manager.split_data_for_validation(os.path.join(file_manager.local_path_training_data, "false"), os.path.join(file_manager.local_path_validation_data, "false"))

# Optional: Disable GPU if desired
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_model(input_shape=(256, 256, 1)):
    return Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

# Subtle data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
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
    EarlyStopping(patience=5, restore_best_weights=True),
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
