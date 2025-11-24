import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
import os

class SentinelDataset(Dataset):
    def __init__(self, image_dir, label_dir, year=None, transform=None):
        """
        Args:
            image_dir (str): Directory with Sentinel-2 images.
            label_dir (str): Directory with Dynamic World labels.
            year (int, optional): Year to filter files by.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        
        all_files = os.listdir(image_dir)
        self.images = sorted([
            f for f in all_files 
            if f.endswith('.tif') and "Sentinel2" in f
        ])
        
        if year is not None:
            self.images = [f for f in self.images if str(year) in f]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Assuming label has same filename or a consistent naming convention
        # Adjust this logic if filenames differ significantly
        label_name = img_name.replace("Sentinel2", "DynamicWorld")
        label_path = os.path.join(self.label_dir, label_name)
        
        with rasterio.open(img_path) as src:
            image = src.read() # (Channels, H, W)
            image = image.astype(np.float32)
            
        with rasterio.open(label_path) as src:
            label = src.read(1) # (H, W)
            label = label.astype(np.int64)
            
        # Normalize image (simple min-max or standardization)
        # Sentinel-2 values are often 0-10000 or 0-1. Adjust based on GEE export.
        # Here assuming 0-1 from GEE script (divide by 10000 done in script)
        image = np.clip(image, 0, 1)
        
        sample = {'image': torch.from_numpy(image), 'label': torch.from_numpy(label)}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

if __name__ == "__main__":
    # Dummy test
    pass
