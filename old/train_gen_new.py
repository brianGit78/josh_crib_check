import os
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from file_sync import FileManager
import creds

def configure_logging(log_dir='logs', log_filename='train_gen.log'):
    """
    Configure logging: writes logs both to a file and console.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        filename=os.path.join(log_dir, log_filename),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(console_handler)


def create_file_manager():
    """
    Initialize and return a FileManager. Also handles file synchronization
    and dataset splitting.
    """
    logging.info('Initializing FileManager')
    file_manager = FileManager(creds.model_name)
    file_manager.create_local_directories()

    logging.info('Syncing source files')
    file_manager.sync_source(creds.nas_user, creds.nas_password, creds.nas_host, creds.nas_path)
    # If you already have the data locally, comment out the sync above.

    logging.info('Splitting data for validation')
    file_manager.split_data_for_validation(
        os.path.join(file_manager.local_path_training_data, "true"),
        os.path.join(file_manager.local_path_validation_data, "true")
    )
    file_manager.split_data_for_validation(
        os.path.join(file_manager.local_path_training_data, "false"),
        os.path.join(file_manager.local_path_validation_data, "false")
    )
    return file_manager

def create_model(input_shape=(256, 256, 1)):
    """
    Build and compile a simple CNN model using Keras Sequential API.
    """
    logging.info('Creating model with input shape %s', input_shape)
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
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    logging.info('Model created and compiled successfully')
    return model

def preprocess_input(img, mask):
    """
    Simple preprocessing function for images.
      - Normalizes the image to [0,1].
      - Applies a grayscale mask to zero out parts of the image that are not needed.
    """
    img = img / 255.0
    img = img * mask
    return img

def load_mask_pil(path, target_size=(256, 256)):
    # Load image with PIL in grayscale
    img = Image.open(path).convert('L')  
    # Resize if needed
    if img.size != target_size:
        img = img.resize(target_size)
    # Convert to numpy, normalize
    arr = np.array(img) / 255.0
    # Expand dims so it becomes (256, 256, 1)
    arr = np.expand_dims(arr, axis=-1)
    return arr

def create_generators(file_manager, mask_path='crib_mask.png'):
    """
    Create training and validation data generators.
    Applies the provided mask as a preprocessing function for training data.
    """
    logging.info('Loading and preprocessing mask')
    # Load the mask as grayscale and normalize
    #mask = load_img(mask_path, color_mode='grayscale', target_size=(256, 256))
    #mask = img_to_array(mask) / 255.0

    mask = load_mask_pil(mask_path)
    logging.info('Mask loaded and preprocessed successfully')

    train_datagen = ImageDataGenerator(
        preprocessing_function=lambda x: preprocess_input(x, mask),
        rotation_range=2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.9, 1.1],
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    target_size = (256, 256)
    batch_size = 64

    train_generator = train_datagen.flow_from_directory(
        file_manager.local_path_training_data,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale'
    )

    logging.info('Training data generator created successfully')
    validation_generator = validation_datagen.flow_from_directory(
        file_manager.local_path_validation_data,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='grayscale'
    )
    logging.info('Validation data generator created successfully')
    return train_generator, validation_generator, train_datagen, mask

def train_and_evaluate(model, train_gen, val_gen, class_weights=None):


    return evaluation

def visualize_mask_application(file_manager, train_datagen, mask, num_samples=3):
    """
    Visualize the mask application process with side-by-side original vs masked images.
    """
    try:
        # 1) Create a separate generator without mask preprocessing
        no_mask_datagen = ImageDataGenerator(rescale=1./255)
        no_mask_generator = no_mask_datagen.flow_from_directory(
            file_manager.local_path_training_data,
            target_size=(256, 256),
            batch_size=num_samples,
            class_mode='binary',
            color_mode='grayscale',
            shuffle=False  # turn off shuffle to keep consistent indexing
        )

        # 2) The existing masked generator (train_datagen might already have the preprocess function)
        #    We also set shuffle=False if we want the images to match index-for-index
        masked_generator = train_datagen.flow_from_directory(
            file_manager.local_path_training_data,
            target_size=(256, 256),
            batch_size=num_samples,
            class_mode='binary',
            color_mode='grayscale',
            shuffle=False
        )

        # 3) Grab a batch from each
        images_unmasked, _ = next(no_mask_generator)
        images_masked, _   = next(masked_generator)

        # Then do the plotting
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))

        for i in range(num_samples):
            # Original
            axes[i, 0].imshow(images_unmasked[i], cmap='gray')
            axes[i, 0].set_title('Original')

            # Mask
            axes[i, 1].imshow(mask[:, :, 0], cmap='gray')
            axes[i, 1].set_title('Mask')

            # Masked result
            axes[i, 2].imshow(images_masked[i], cmap='gray')
            axes[i, 2].set_title('Masked')

        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        viz_path = os.path.join('logs', f'mask_viz_{timestamp}.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Mask visualization saved to {viz_path}")

    except Exception as e:
        logging.error(f"Error during visualization: {str(e)}")
        return
    
def compute_class_weights(file_manager):
    # Example: compute class weights
    false_dir = os.path.join(file_manager.local_path_training_data, 'false')
    true_dir  = os.path.join(file_manager.local_path_training_data, 'true')
    
    num_false = len(os.listdir(false_dir))
    num_true  = len(os.listdir(true_dir))
    total     = num_false + num_true

    weight_for_false = total / (2.0 * num_false)  # or use a custom ratio
    weight_for_true  = total / (2.0 * num_true)

    class_weights = {
        0: weight_for_false,
        1: weight_for_true
    }
    logging.info(f"Using class weights: {class_weights}")

    return class_weights

def main():
    configure_logging()
    file_manager = create_file_manager()

    # Create the model
    model = create_model()

    #create the class weights
    class_weights = compute_class_weights(file_manager)

    # Create the data generators
    train_generator, validation_generator, train_datagen, mask = create_generators(file_manager)

    #validate mask application
    #visualize_mask_application(file_manager, train_datagen, mask)
    """
    Train the model with EarlyStopping and ModelCheckpoint.
    Evaluate on validation data, then return the final evaluation metrics.
    """
    # Create a timestamped log directory for TensorBoard - tensorboard --logdir logs/tensorboard
    # C:\Python312\python.exe C:\Users\<user>\AppData\Roaming\Python\Python312\site-packages\tensorboard\main.py --logdir=C:\Users\<user>\josh_crib_check\logs\tensorboard
    log_dir = os.path.join("logs", "tensorboard", datetime.now().strftime("%Y%m%d-%H%M%S"))
    #tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

    epochs=50
    #model_filename='best_model.keras'
    best_model_path = 'best_model.keras'

    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            best_model_path,
            save_best_only=True
        ),
        #tb_callback
    ]

    logging.info('Starting model training')
    model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    logging.info('Evaluating model')
    evaluation = model.evaluate(validation_generator)
    logging.info('Validation Loss: %.4f, Validation Accuracy: %.4f', evaluation[0], evaluation[1])
    print(f"Validation Loss: {evaluation[0]:.4f}, Validation Accuracy: {evaluation[1]:.4f}")

    # Train and evaluate
    #train_and_evaluate(model, train_generator, validation_generator, class_weights=class_weights)

    # Remove old model file if it exists
    logging.info('Archiving old model file if any')
    file_manager.archive_model_file()

    # Load the best model weights and save the final model
    
    if os.path.exists(best_model_path):
        logging.info('Loading best weights from checkpoint')
        model.load_weights(best_model_path)

    final_model_path = file_manager.model_file_path
    logging.info('Saving final model to %s', final_model_path)
    model.save(final_model_path)

if __name__ == '__main__':
    main()

