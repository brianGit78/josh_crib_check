import argparse
import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch import amp
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from device_utils import describe_device, select_device
from file_sync import FileManager
from preprocessing import build_transforms
from pt_cnn import CribMobileNet
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

def train_model(model, train_loader, val_loader, device, num_epochs=50, patience=5):
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # Reduce LR by half when plateau is detected
        patience=3,  # Wait 3 epochs before reducing
    )

    use_autocast = device.type in {"cuda", "mps"}
    scaler = amp.GradScaler("cuda") if device.type == "cuda" else amp.GradScaler(enabled=False)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        epoch_train_start_time = time.time()
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            with amp.autocast(
                device_type=device.type if use_autocast else "cpu",
                dtype=torch.float16 if device.type == "cuda" else None,
                enabled=use_autocast,
            ):
                logits = model(images)
                loss = criterion(logits.squeeze(), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

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
        
        scheduler.step(epoch_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logging.info("Current learning rate: %s", current_lr)

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

def main():
    configure_logging()
    file_manager = create_file_manager()

    transforms_start_time = time.time()
    train_transforms = build_transforms('crib_mask.png', train=True)
    val_transforms = build_transforms('crib_mask.png', train=False)

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
    num_workers = min(8, (os.cpu_count() or 2))

    device = select_device()
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision('high')

    pin_memory = device.type == 'cuda'

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    
    transforms_end_time = time.time()
    logging.info(f"Transforms defined - total time taken: {transforms_end_time - transforms_start_time:.2f} seconds")

    model = CribMobileNet(pretrained=True, dropout=0.35)

    logging.info(f"Training on {len(train_dataset)} samples")
    print(f"Using device: {describe_device(device)}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if device.type == 'cuda':
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current device index: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    trained_model = train_model(model, train_loader, val_loader, device, num_epochs=50, patience=5)
    torch.save(trained_model.state_dict(), file_manager.model_file_path)

    model_train_end_time = time.time()
    logging.info(f"Training complete - total time taken: {model_train_end_time - model_train_start_time:.2f} seconds")
    # LOAD BEST MODEL
    trained_model.load_state_dict(torch.load(file_manager.model_file_path))
    trained_model.eval()

if __name__ == '__main__':
    main()
