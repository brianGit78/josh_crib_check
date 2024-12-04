import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from file_sync import FileManager
import creds

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

# Data generator for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    file_manager.local_path_training_data,  # path to classified training images
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Data generator for validation
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    file_manager.local_path_validation_data,  # Path to your validation data
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Define the model
model = Sequential([
    Input(shape=(150, 150, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the modelServer12345#

# Define early stopping callback
# this prevents the model from overfitting by monitoring the validation loss if it does not improve in 5 passes
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(
    train_generator,
    epochs=50,  # Set a high number of epochs
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model
evaluation = model.evaluate(validation_generator)
print(f"Validation Loss: {evaluation[0]}, Validation Accuracy: {evaluation[1]}")

model.save(file_manager.model_file_path)  # Save the trained model in the native Keras format
