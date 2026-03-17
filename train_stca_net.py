import os
import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
from tqdm import tqdm
import random
import numpy as np

from models.stca_net import STCANet


class JPEGCompression:
    """Custom transform: simulate JPEG compression artifacts."""
    def __init__(self, quality_range=(30, 95)):
        self.quality_range = quality_range

    def __call__(self, img):
        if random.random() < 0.3:
            import io
            quality = random.randint(*self.quality_range)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img = Image.open(buffer).convert('RGB')
        return img


class GaussianNoise:
    """Custom transform: add Gaussian noise to simulate camera noise."""
    def __init__(self, mean=0, std_range=(0.01, 0.05)):
        self.mean = mean
        self.std_range = std_range

    def __call__(self, tensor):
        if random.random() < 0.3:
            std = random.uniform(*self.std_range)
            noise = torch.randn_like(tensor) * std + self.mean
            tensor = tensor + noise
            tensor = torch.clamp(tensor, 0.0, 1.0)
        return tensor


class RandomGaussianBlur:
    """Custom transform: apply Gaussian blur to simulate out-of-focus/recompression."""
    def __init__(self, radius_range=(0.5, 2.0)):
        self.radius_range = radius_range

    def __call__(self, img):
        if random.random() < 0.2:
            radius = random.uniform(*self.radius_range)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class SimpleDeepfakeDataset(Dataset):
    """
    Loads images from a folder structure:
    root_dir/
      real/
        img1.jpg
        ...
      fake/
        img1.jpg
        ...
    """
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        real_dir = os.path.join(root_dir, 'real')
        fake_dir = os.path.join(root_dir, 'fake')
        
        # Class 0: Fake, Class 1: Real
        fake_limit = max_samples // 2 if max_samples else None
        real_limit = max_samples // 2 if max_samples else None
        self._load_class(fake_dir, 0, fake_limit)
        self._load_class(real_dir, 1, real_limit)
        
        # Shuffle to mix classes
        random.shuffle(self.samples)
            
    def _load_class(self, dir_path, label, limit):
        if not os.path.exists(dir_path):
            print(f"Warning: Directory not found - {dir_path}")
            return
            
        count = 0
        files = sorted(os.listdir(dir_path))
        for fname in files:
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.samples.append((os.path.join(dir_path, fname), label))
                count += 1
                if limit and count >= limit:
                    break
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            # Return a blank tensor and the label if image is corrupted
            print(f"Error loading {img_path}: {e}")
            return torch.zeros((3, 224, 224)), label


def get_transforms():
    """
    Enhanced augmentation pipeline:
    - JPEG compression simulation
    - Gaussian blur
    - Gaussian noise
    - Standard flips/rotation/color jitter
    """
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        JPEGCompression(quality_range=(30, 95)),        # Simulate compression
        RandomGaussianBlur(radius_range=(0.5, 2.0)),    # Simulate blur
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        GaussianNoise(std_range=(0.01, 0.05)),           # Simulate sensor noise
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_path):
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        
        # Training Phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        train_bar = tqdm(train_loader, desc='Training')
        for inputs, labels in train_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # STCA-Net returns output and attention_weights
            outputs, _ = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        history['train_loss'].append(round(epoch_loss, 4))
        history['train_acc'].append(round(epoch_acc.item(), 4))
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
        # Validation Phase
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_corrects = 0
            
            val_bar = tqdm(val_loader, desc='Validation')
            with torch.no_grad():
                for inputs, labels in val_bar:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    outputs, _ = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels.data)
                    
            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_epoch_acc = val_corrects.double() / len(val_loader.dataset)
            history['val_loss'].append(round(val_epoch_loss, 4))
            history['val_acc'].append(round(val_epoch_acc.item(), 4))
            print(f'Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}')
            
            # Step the scheduler based on validation loss
            if scheduler:
                scheduler.step(val_epoch_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Current LR: {current_lr:.6f}')
            
            if val_epoch_acc > best_acc:
                best_acc = val_epoch_acc
                torch.save(model.state_dict(), save_path)
                print(f"⭐ New best model saved to {save_path}!")
        else:
            # If no validation set, save every epoch
            torch.save(model.state_dict(), save_path)
            
    print('\n' + '=' * 40)
    print('Training complete.')
    if val_loader:
        print(f'Best Val Acc: {best_acc:.4f}')
    
    # Save training history
    history_path = os.path.join(os.path.dirname(save_path), 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train STCA-Net for Deepfake Detection")
    parser.add_argument('--dataset', type=str, default='dataset/140k', help='Path to dataset directory containing real/fake folders')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--samples', type=int, default=10000, help='Max samples to load (0 = all)')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--unfreeze-layers', type=int, default=3, help='Number of MobileNet layers to unfreeze from the end')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_transform, val_transform = get_transforms()
    
    max_samples = args.samples if args.samples > 0 else None
    print(f"Loading dataset from: {args.dataset} (Max {max_samples or 'ALL'} samples)...")
    dataset = SimpleDeepfakeDataset(args.dataset, transform=train_transform, max_samples=max_samples)
    
    if len(dataset) == 0:
        print("Error: Dataset is empty. Check your paths.")
        exit(1)
    
    # Count class distribution
    fake_count = sum(1 for _, label in dataset.samples if label == 0)
    real_count = sum(1 for _, label in dataset.samples if label == 1)
    print(f"Loaded {len(dataset)} images (Fake: {fake_count}, Real: {real_count}).")
    
    # 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create a copy of the dataset for validation with different transforms
    # Note: random_split shares the underlying dataset, so we use a wrapper
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print("Initializing STCA-Net architecture...")
    model = STCANet().to(device)
    
    # Unfreeze last N layers of MobileNet for domain adaptation
    # MobileNet features has ~13 sequential blocks
    all_layers = list(model.spatial_extractor.children())
    num_layers = len(all_layers)
    
    # First, freeze everything
    for param in model.spatial_extractor.parameters():
        param.requires_grad = False
    
    # Then unfreeze the last N layers
    layers_to_unfreeze = min(args.unfreeze_layers, num_layers)
    for layer in all_layers[-layers_to_unfreeze:]:
        for param in layer.parameters():
            param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Unfroze last {layers_to_unfreeze} MobileNet layers. "
          f"Trainable: {trainable_params:,} / {total_params:,} parameters")
    
    # Loss with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Use AdamW (weight decay regularization)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    
    # Learning rate scheduler — reduce LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    save_dir = "model"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "stca_net_weights.pt")
    
    print("\nTraining Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Label Smoothing: {args.label_smoothing}")
    print(f"  Unfrozen Layers: {layers_to_unfreeze}")
    print(f"  Optimizer: AdamW (weight_decay=0.01)")
    print(f"  Scheduler: ReduceLROnPlateau (factor=0.5, patience=3)")
    print(f"  Save Path: {save_path}")
    print()
    
    print("Starting training loop...")
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, args.epochs, device, save_path)
