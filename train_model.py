import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pt_cnn import SimpleCNN
from PIL import Image
import numpy as np

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
    #file_manager.sync_source(creds.nas_user, creds.nas_password, creds.nas_host, creds.nas_path)
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

def main():
    configure_logging()
    file_manager = create_file_manager()
    
    train_transforms = T.Compose([
        T.Grayscale(num_output_channels=1),  # ensure 1-channel grayscale
        T.Resize((256, 256)),
        T.RandomRotation(degrees=2),
        #T.RandomResizedCrop(size=256, scale=(0.9, 1.0)),  # approximate zoom
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), shear=0.05), # this or above
        T.RandomHorizontalFlip(p=0.5),        
        T.ColorJitter(brightness=(0.9, 1.1)),  # approx brightness_range
        T.ToTensor(),  # convert PIL image [0,255] -> torch tensor [0,1]
        #ApplyMask('crib_mask.png', target_size=(256, 256))  # apply custom mask
        ApplyMask('crib_mask.png')
    ])

    # For validation, typically fewer or no augmentations
    val_transforms = T.Compose([
        T.Grayscale(num_output_channels=1),
        T.Resize((256, 256)),
        T.ToTensor()
    ])

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

    model = SimpleCNN()

    # 5) Training setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 6) Training loop with early stopping
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 5
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # TRAIN
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)

        # VALIDATION
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                logits = model(images)
                loss = criterion(logits.squeeze(), labels)
                val_loss += loss.item() * images.size(0)

                preds = torch.sigmoid(logits)
                preds = (preds > 0.5).float()
                correct += (preds.squeeze() == labels).sum().item()
                total += labels.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total

        print(f"Epoch {epoch+1}/{num_epochs} "
            f"Train Loss: {epoch_train_loss:.4f}, "
            f"Val Loss: {epoch_val_loss:.4f}, "
            f"Val Acc: {val_accuracy:.4f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), file_manager.model_file_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # LOAD BEST MODEL
    model.load_state_dict(torch.load(file_manager.model_file_path))
    model.eval()

if __name__ == '__main__':
    main()
