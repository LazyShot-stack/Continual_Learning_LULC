"""
Inference script for trained LULC models.
Loads a checkpoint and generates classification maps.
"""

import torch
import torch.nn.functional as F
import rasterio
import numpy as np
import os
import argparse
from pathlib import Path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network import UrbanMapper

def load_model(checkpoint_path, num_classes=9, device='cpu'):
    """Load trained model from checkpoint."""
    model = UrbanMapper(num_classes=num_classes, num_channels=4)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, device='cpu', output_size=None):
    """
    Predict classification for a single image.
    
    Args:
        model: Trained UrbanMapper model
        image_path: Path to Sentinel-2 GeoTIFF (4 channels)
        device: 'cpu' or 'cuda'
        output_size: Tuple (H, W) to resize output. If None, matches input.
    
    Returns:
        predictions: numpy array of shape (H, W) with class indices
        probabilities: numpy array of shape (9, H, W) with class probabilities
    """
    
    # Load image
    with rasterio.open(image_path) as src:
        image = src.read()  # (4, H, W)
        image = image.astype(np.float32)
        image_profile = src.profile
    
    # Normalize
    image = np.clip(image, 0, 1)
    
    # Convert to tensor and add batch dimension
    image_tensor = torch.from_numpy(image).unsqueeze(0).to(device)  # (1, 4, H, W)
    
    # Forward pass
    with torch.no_grad():
        output = model(image_tensor)  # (1, 9, H, W)
        
        # Resize if needed
        if output_size is not None:
            output = F.interpolate(output, size=output_size, mode='bilinear', align_corners=False)
        
        # Get probabilities and predictions
        probabilities = F.softmax(output, dim=1)  # (1, 9, H, W)
        predictions = torch.argmax(output, dim=1).squeeze(0)  # (H, W)
    
    return predictions.cpu().numpy(), probabilities.squeeze(0).cpu().numpy(), image_profile

def save_prediction(predictions, output_path, profile):
    """Save predictions as GeoTIFF."""
    profile.update(count=1, dtype=rasterio.uint8, nodata=255)
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(predictions.astype(np.uint8), 1)

def save_probabilities(probabilities, output_path, profile):
    """Save class probabilities as multi-band GeoTIFF."""
    num_classes = probabilities.shape[0]
    profile.update(count=num_classes, dtype=rasterio.float32)
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        for i in range(num_classes):
            dst.write((probabilities[i] * 255).astype(np.float32), i + 1)

def main():
    parser = argparse.ArgumentParser(description='Generate LULC classification maps')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to model checkpoint (e.g., model/checkpoint_2020.pth)')
    parser.add_argument('--image', type=str, required=True, 
                        help='Path to Sentinel-2 image (GeoTIFF with 4 bands)')
    parser.add_argument('--output', type=str, default='output/prediction.tif',
                        help='Path to save classification map')
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Save class probabilities to separate file')
    parser.add_argument('--num_classes', type=int, default=9,
                        help='Number of classes (default: 9 for Dynamic World)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return
    
    if not os.path.exists(args.image):
        print(f"Error: Image not found: {args.image}")
        return
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, num_classes=args.num_classes, device=device)
    
    # Predict
    print(f"Processing image: {args.image}...")
    predictions, probabilities, profile = predict_image(model, args.image, device=device)
    
    # Save results
    print(f"Saving classification map to {args.output}...")
    save_prediction(predictions, args.output, profile)
    
    if args.save_probabilities:
        prob_output = args.output.replace('.tif', '_probabilities.tif')
        print(f"Saving probabilities to {prob_output}...")
        save_probabilities(probabilities, prob_output, profile)
    
    # Statistics
    unique_classes, counts = np.unique(predictions, return_counts=True)
    print("\nClassification Summary:")
    print(f"Image size: {predictions.shape}")
    print(f"Classes found: {unique_classes}")
    print(f"Pixel counts: {counts}")
    print(f"Class percentages:")
    for cls, cnt in zip(unique_classes, counts):
        pct = 100 * cnt / predictions.size
        print(f"  Class {cls}: {pct:.2f}%")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
