
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from network import UrbanMapper
from dataset import SentinelDataset
from ewc import EWC

def train_task(model, train_loader, ewc, epochs, device, learning_rate=1e-3, ewc_lambda=0.4):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    model.train()
    
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
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}")
        
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Model
    # In a real scenario, we would load the previous year's checkpoint if it exists
    model = UrbanMapper(num_classes=9, num_channels=4).to(device)
    
    checkpoint_path = f"model/checkpoint_{target_year - 1}.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))
    
    print(f"--- Training on Year {target_year} ---")
    
    if not os.path.exists(data_root):
        print(f"Data directory {data_root} not found. Aborting.")
        return

    # Placeholder for dataset loading
    # In reality, filter by year
    dataset = SentinelDataset(data_root, data_root, year=target_year) 
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
    
    # Load EWC if previous model existed (simplified)
    ewc = None
    if os.path.exists(checkpoint_path):
        # We would need to load the Fisher matrix too, or recalculate it from a subset of old data
        # For this demo, we skip loading EWC state and just show the training call
        pass

    # Train
    model = train_task(model, train_loader, ewc, epochs=5, device=device)
    
    # Calculate Fisher for next time
    print("Calculating Fisher Information for EWC...")
    ewc = EWC(model, train_loader)
    
    # Save model checkpoint
    save_path = f"model/checkpoint_{target_year}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
