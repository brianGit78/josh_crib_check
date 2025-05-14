import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pt_cnn import SimpleCNN
from PIL import Image
import numpy as np

from file_sync import FileManager
import creds

parser = argparse.ArgumentParser(description="Training script")
parser.add_argument("--skip_source_sync", action="store_true", help="Skip source sync")
args = parser.parse_args()

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

    if not args.skip_source_sync:
        file_sync_start_time = time.time()
        logging.info('Syncing source files')
        file_manager.sync_source(creds.nas_user, creds.nas_password, creds.nas_host, creds.nas_path)

        logging.info('Copying static validation images')
        file_manager.copy_static_validation_data(
            creds.nas_user,
            creds.nas_password,
            creds.nas_host,
            creds.static_validation_path
        )

        logging.info('Splitting data ramdomly for validation')
        file_manager.split_data_for_validation(
            os.path.join(file_manager.local_path_training_data, "true"),
            os.path.join(file_manager.local_path_validation_data, "true")
        )
        file_manager.split_data_for_validation(
            os.path.join(file_manager.local_path_training_data, "false"),
            os.path.join(file_manager.local_path_validation_data, "false")
        )


        file_sync_end_time = time.time()
        logging.info(f'Source sync and data split took {file_sync_end_time - file_sync_start_time:.2f} seconds')

    return file_manager

class ApplyMask:
    def __init__(self, mask_path, target_size=(256, 256)):
        # Load the mask once
        mask_pil = Image.open(mask_path).convert('L').resize(target_size)
        mask_array = np.array(mask_pil) / 255.0
        self.mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).float()  
        # unsqueeze(0) -> shape becomes (1, H, W) for grayscale

    def __call__(self, img):
        # img is a tensor in [0,1], shape (1, H, W) for grayscale
        return img * self.mask_tensor
    
def train_model(model, train_loader, val_loader, device, num_epochs=50, patience=5):
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        epoch_train_start_time = time.time()
        model.train()
        running_loss = 0.0

        # --- TRAIN LOOP ---
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            logits = model(images)             # shape: (batch_size, 1)
            loss = criterion(logits.squeeze(), labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device)

                logits = model(images)
                loss = criterion(logits.squeeze(), labels)
                val_loss += loss.item() * images.size(0)

                # If you want accuracy:
                preds = torch.sigmoid(logits)              # convert logits -> probabilities
                preds = (preds > 0.5).float()              # threshold at 0.5
                correct += (preds.squeeze() == labels).sum().item()
                total += labels.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total

        # Print or log training/val metrics
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_acc:.4f}")

        # --- Early Stopping or Checkpointing ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            # Save best model
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        epoch_train_end_time = time.time()
        logging.info(f"Epoch {epoch+1} complete - total time taken: {epoch_train_end_time - epoch_train_start_time:.2f} seconds")

    logging.info("Loading best model weights.")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    return model

class HistEqualization:
    def __call__(self, img):
        # Pillow-based histogram equalization
        return TF.equalize(img)

def main():
    configure_logging()
    file_manager = create_file_manager()
    
    # Define transforms for training and validation
    transforms_start_time = time.time()
    train_transforms = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((256, 256)),
        HistEqualization(),
        T.RandomAffine(
            degrees=2,             # ±2 degrees
            translate=(0.05, 0.05),# ±5% shift
            scale=(0.9, 1.1),      # ±10% zoom
            shear=(-3, 3),         # ~±3 degrees shear (approx 0.05 rad)
            interpolation=InterpolationMode.NEAREST,
            fill=0                 # fill black for "empty" pixels
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=(0.9, 1.1)),
        T.ToTensor(),
        ApplyMask('crib_mask.png')  # your custom mask transform
    ])

    # For validation, typically fewer or no augmentations
    val_transforms = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((256, 256)),
        HistEqualization(),
        T.ToTensor()
    ])

    model_train_start_time = time.time()
    train_dataset = ImageFolder(
        root=file_manager.local_path_training_data,
        transform=train_transforms
    )

    val_dataset = ImageFolder(
        root=file_manager.local_path_validation_data,
        transform=val_transforms
    )

    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    transforms_end_time = time.time()
    logging.info(f"Transforms defined - total time taken: {transforms_end_time - transforms_start_time:.2f} seconds")

    model = SimpleCNN()

    logging.info(f"Training on {len(train_dataset)} samples")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #make sure we are using CUDA
    print(f"Using device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    trained_model = train_model(model, train_loader, val_loader, device, num_epochs=50, patience=5)
    torch.save(trained_model.state_dict(), file_manager.model_file_path)

    model_train_end_time = time.time()
    logging.info(f"Training complete - total time taken: {model_train_end_time - model_train_start_time:.2f} seconds")
    # LOAD BEST MODEL
    trained_model.load_state_dict(torch.load(file_manager.model_file_path))
    trained_model.eval()

if __name__ == '__main__':
    main()

