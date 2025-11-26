"""
Validation script to evaluate model performance on test data.
Computes metrics like accuracy, precision, recall, F1-score, IoU.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, jaccard_score
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from network import UrbanMapper
from dataset import SentinelDataset
import argparse

def compute_metrics(predictions, targets, num_classes=9):
    """Compute evaluation metrics."""
    # Flatten
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Accuracy
    accuracy = accuracy_score(targets, predictions)
    
    # Per-class metrics
    precision = precision_score(targets, predictions, average='weighted', zero_division=0)
    recall = recall_score(targets, predictions, average='weighted', zero_division=0)
    f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    # Mean IoU
    iou_per_class = []
    for cls in range(num_classes):
        try:
            iou = jaccard_score(targets == cls, predictions == cls, zero_division=0)
            iou_per_class.append(iou)
        except:
            iou_per_class.append(0)
    
    miou = np.mean(iou_per_class)
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions, labels=range(num_classes))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'miou': miou,
        'iou_per_class': iou_per_class,
        'confusion_matrix': cm
    }

def validate(model, test_loader, device, num_classes=9):
    """Validate model on test dataset."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['image'].to(device)
            targets = batch['label'].to(device)
            
            outputs = model(inputs)
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    metrics = compute_metrics(all_predictions, all_targets, num_classes)
    return metrics, all_predictions, all_targets

def main():
    parser = argparse.ArgumentParser(description='Validate LULC model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to test data directory')
    parser.add_argument('--year', type=int, required=True,
                        help='Year to validate')
    parser.add_argument('--num_classes', type=int, default=9,
                        help='Number of classes')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = UrbanMapper(num_classes=args.num_classes).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    # Load dataset
    dataset = SentinelDataset(args.data_dir, args.data_dir, year=args.year)
    test_loader = DataLoader(dataset, batch_size=4)
    
    print(f"Validating on {len(dataset)} samples...")
    
    # Validate
    metrics, predictions, targets = validate(model, test_loader, device, args.num_classes)
    
    # Print results
    print("\n=== Validation Results ===")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"mIoU:      {metrics['miou']:.4f}")
    
    print("\nPer-class IoU:")
    for i, iou in enumerate(metrics['iou_per_class']):
        print(f"  Class {i}: {iou:.4f}")
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
    
    # Save results
    np.save('validation_predictions.npy', predictions)
    np.save('validation_targets.npy', targets)
    print("\nPredictions and targets saved to .npy files")

if __name__ == "__main__":
    main()
