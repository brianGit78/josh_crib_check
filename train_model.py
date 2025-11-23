import argparse
import json
import logging
import os
import time
import hashlib
from collections import Counter

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


def _hash_file(path: str) -> str:
    # Use a content hash (SHA-1) so duplicate images with different filenames
    # collapse to the same key. Reading in 8 KB chunks keeps memory usage low
    # even for large files.
    hasher = hashlib.sha1()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def deduplicate_imagefolder(dataset: ImageFolder, split_name: str) -> None:
    """Remove exact-duplicate images (by content hash) to avoid overweighting repeated snapshots."""

    if not dataset.samples:
        return

    unique_samples = []
    hash_to_label = {}
    conflicts = 0

    for path, label in dataset.samples:
        file_hash = _hash_file(path)
        # First time we see a hash, remember its label and keep the sample.
        if file_hash not in hash_to_label:
            hash_to_label[file_hash] = label
            unique_samples.append((path, label))
        # If the same file content shows up under a different label, surface it
        # as a potential labeling mistake instead of silently choosing one.
        elif hash_to_label[file_hash] != label:
            conflicts += 1
            logging.warning('Duplicate content with conflicting labels detected in %s: %s', split_name, path)

    removed = len(dataset.samples) - len(unique_samples)
    if removed > 0:
        logging.info(
            'Deduplicated %s split: removed %d exact duplicates (kept %d unique samples)',
            split_name,
            removed,
            len(unique_samples)
        )
    if conflicts:
        logging.warning('Found %d duplicate files with conflicting labels in %s; please double-check labeling.', conflicts, split_name)

    dataset.samples = unique_samples
    dataset.imgs = unique_samples
    dataset.targets = [label for _, label in unique_samples]


def build_balanced_sampler(dataset: ImageFolder):
    """Create a class-balanced sampler to reduce bias from uneven or duplicate-heavy folders."""

    if not dataset.targets:
        return None

    class_counts = Counter(dataset.targets)
    num_samples = len(dataset.targets)
    class_weights = {cls: num_samples / (len(class_counts) * count) for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in dataset.targets]
    return torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=num_samples, replacement=True)


def log_split_stats(dataset: ImageFolder, split_name: str) -> None:
    if not dataset.targets:
        logging.warning('No samples found in %s split after deduplication', split_name)
        return

    counts = Counter(dataset.targets)
    label_names = {idx: name for idx, name in enumerate(dataset.classes)}
    readable_counts = {label_names[idx]: count for idx, count in counts.items()}
    logging.info('%s split class distribution: %s', split_name, readable_counts)


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


def compute_validation_predictions(model, val_loader, device):
    model.eval()
    probabilities = []
    labels_list = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits.squeeze())
            probabilities.append(probs.detach().cpu())
            labels_list.append(labels.float().cpu())

    return torch.cat(probabilities), torch.cat(labels_list)


def find_best_threshold(probabilities, labels, beta=0.5, target_precision=0.995):
    # Sweep a fine-grained set of thresholds toward the high-confidence side to
    # prioritize precision and cut false positives.
    thresholds = torch.linspace(0.01, 0.99, steps=99)
    best = {"threshold": 0.5, "fscore": 0.0, "precision": 0.0, "recall": 0.0}
    best_high_precision = None

    for threshold in thresholds:
        preds = (probabilities >= threshold).float()

        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        fscore = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-8)

        if precision >= target_precision:
            if best_high_precision is None or recall > best_high_precision["recall"]:
                best_high_precision = {
                    "threshold": threshold.item(),
                    "fscore": fscore,
                    "precision": precision,
                    "recall": recall,
                }

        if fscore > best["fscore"]:
            best.update(
                {
                    "threshold": threshold.item(),
                    "fscore": fscore,
                    "precision": precision,
                    "recall": recall,
                }
            )

    # Prefer the highest-recall option that satisfies the precision target.
    return best_high_precision or best


def save_thresholds(file_manager, base_threshold, best_metrics, margin=0.05):
    # Shrink the hysteresis window as thresholds move toward 1.0 so "on" and
    # "off" remain separated without letting borderline highs trigger.
    adaptive_margin = margin * (1.0 - base_threshold + 0.2)
    threshold_on = min(0.995, base_threshold + adaptive_margin)
    threshold_off = max(0.01, base_threshold - adaptive_margin)
    payload = {
        "base_threshold": base_threshold,
        "threshold_on": threshold_on,
        "threshold_off": threshold_off,
        "calibration": best_metrics,
    }

    with open(file_manager.thresholds_file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logging.info(
        "Saved calibrated thresholds to %s (on=%.3f, off=%.3f)",
        file_manager.thresholds_file_path,
        threshold_on,
        threshold_off,
    )

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


def calibrate_thresholds(model, val_loader, device, file_manager):
    probabilities, labels = compute_validation_predictions(model, val_loader, device)
    best_metrics = find_best_threshold(probabilities, labels)
    save_thresholds(file_manager, best_metrics["threshold"], best_metrics)
    logging.info(
        "Calibration summary - threshold: %.3f | f_beta: %.4f | precision: %.4f | recall: %.4f",
        best_metrics["threshold"],
        best_metrics["fscore"],
        best_metrics["precision"],
        best_metrics["recall"],
    )
    logging.info(
        "Precision-targeted calibration chooses the highest recall with >=0.995 precision; rerun calibration if lighting/context shifts."
    )

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
    deduplicate_imagefolder(train_dataset, 'train')
    train_sampler = build_balanced_sampler(train_dataset)
    log_split_stats(train_dataset, 'train')

    val_dataset = ImageFolder(
        root=file_manager.local_path_validation_data,
        transform=val_transforms
    )
    deduplicate_imagefolder(val_dataset, 'val')
    log_split_stats(val_dataset, 'val')

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
        shuffle=train_sampler is None,
        sampler=train_sampler,
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
    calibrate_thresholds(trained_model, val_loader, device, file_manager)
    torch.save(trained_model.state_dict(), file_manager.model_file_path)

    model_train_end_time = time.time()
    logging.info(f"Training complete - total time taken: {model_train_end_time - model_train_start_time:.2f} seconds")
    # LOAD BEST MODEL
    trained_model.load_state_dict(torch.load(file_manager.model_file_path))
    trained_model.eval()

if __name__ == '__main__':
    main()
