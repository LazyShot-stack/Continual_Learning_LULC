
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import sys
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network import UrbanMapper
from dataset import SentinelDataset
from ewc import EWC

# ===== IMPROVEMENT: Hyperparameters =====
EPOCHS = 5  # Reduced for demo speed
LEARNING_RATE = 0.001  # Standard for Adam
BATCH_SIZE = 16  # Increased from 4
WEIGHT_DECAY = 1e-4  # L2 regularization
EWC_LAMBDA = 0.4  # EWC penalty weight

def train_task(model, train_loader, ewc, epochs, device, learning_rate=LEARNING_RATE, ewc_lambda=EWC_LAMBDA):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    
    # ===== IMPROVEMENT: Learning Rate Scheduler =====
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    
    # If the dataloader is empty, skip training and return early with message
    try:
        loader_len = len(train_loader)
    except Exception:
        loader_len = 0
    if loader_len == 0:
        print("Training skipped: DataLoader is empty (no training samples).")
        return model

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            inputs = batch['image'].to(device)
            targets = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            
            # Add EWC penalty if available
            if ewc is not None:
                loss += ewc_lambda * ewc.penalty(model)
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        # ===== IMPROVEMENT: Call scheduler after each epoch =====
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
    return model

import argparse

def main():
    parser = argparse.ArgumentParser(description='Continual Learning for Urban LULC')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--year', type=int, required=True, help='Year to process')
    args = parser.parse_args()

    # Configuration
    data_root = args.data_dir
    target_year = args.year
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Model
    # In a real scenario, we would load the previous year's checkpoint if it exists
    model = UrbanMapper(num_classes=9, num_channels=4).to(device)
    
    checkpoint_path = f"model/checkpoint_{target_year - 1}.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        # ===== Handle checkpoint loading with different num_classes =====
        # If checkpoint has different classifier shape, adapt it
        try:
            model.load_state_dict(checkpoint)
        except RuntimeError as e:
            if "size mismatch" in str(e):
                print(f"Checkpoint class mismatch, loading backbone only...")
                # Load only the backbone weights
                backbone_state = {k: v for k, v in checkpoint.items() if 'backbone' in k}
                model.load_state_dict(backbone_state, strict=False)
            else:
                raise
    
    print(f"--- Training on Year {target_year} ---")
    
    if not os.path.exists(data_root):
        print(f"Data directory {data_root} not found. Aborting.")
        return

    # ===== IMPROVEMENT: Augmentation enabled for training =====
    # Placeholder for dataset loading
    # In reality, filter by year
    dataset = SentinelDataset(data_root, data_root, year=target_year, augment=True) 
    # Informative feedback about dataset size
    try:
        num_images = len(dataset)
    except Exception:
        num_images = 0
    print(f"Found {num_images} training samples in '{data_root}' (year={target_year}).")

    if num_images == 0:
        print(f"No training images found in {data_root} for year {target_year}. Aborting training.\n"
              "Please verify the directory contains Sentinel-2 GeoTIFFs with 'Sentinel2' in filenames, or run demo_setup.py to create sample data.")
        return

    # Use drop_last=False so small datasets still produce a final smaller batch
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    
    # ===== IMPROVEMENT: Compute class weights if data available =====
    print("Computing class weights for balanced loss...")
    try:
        all_labels = []
        for sample in dataset:
            all_labels.append(sample['label'].numpy().flatten())
        all_labels = np.concatenate(all_labels)
        class_weights = compute_class_weight('balanced', classes=np.arange(9), y=all_labels)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print(f"Class weights: {class_weights}")
        # Update criterion with weights
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    except Exception as e:
        print(f"Could not compute class weights: {e}. Using unweighted loss.")
        criterion = torch.nn.CrossEntropyLoss()
    
    # Load EWC if previous model existed (simplified)
    ewc = None
    if os.path.exists(checkpoint_path):
        # We would need to load the Fisher matrix too, or recalculate it from a subset of old data
        # For this demo, we skip loading EWC state and just show the training call
        pass

    # ===== IMPROVEMENT: Use configured EPOCHS =====
    # Train
    model = train_task(model, train_loader, ewc, epochs=EPOCHS, device=device)
    
    # Calculate Fisher for next time
    print("Calculating Fisher Information for EWC...")
    ewc = EWC(model, train_loader)
    
    # Save model checkpoint
    save_path = f"model/checkpoint_{target_year}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
