import os
import numpy as np
import rasterio
from rasterio.transform import from_origin

def create_dummy_tiff(path, channels=4, size=(256, 256), is_label=False):
    """
    Creates a dummy GeoTIFF file.
    """
    transform = from_origin(77.5, 13.0, 0.0001, 0.0001)
    
    if is_label:
        # Labels: 0-8 classes
        data = np.random.randint(0, 9, size=(1, size[0], size[1])).astype(np.uint8)
        dtype = rasterio.uint8
        count = 1
    else:
        # Sentinel-2: 4 channels, float 0-1
        data = np.random.rand(channels, size[0], size[1]).astype(np.float32)
        dtype = rasterio.float32
        count = channels

    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=size[0],
        width=size[1],
        count=count,
        dtype=dtype,
        crs='+proj=latlong',
        transform=transform,
    ) as dst:
        dst.write(data)

def main():
    base_dir = "LULC_Continual_Learning_Data"
    os.makedirs(base_dir, exist_ok=True)
    
    years = [2018, 2019, 2020]
    
    print(f"Generating synthetic data in {base_dir}...")
    
    for year in years:
        for i in range(4):
            # Create dummy Sentinel-2 image
            s2_name = f"Sentinel2_{year}_{i}.tif"
            create_dummy_tiff(os.path.join(base_dir, s2_name), channels=4)
            
            # Create dummy Dynamic World label
            dw_name = f"DynamicWorld_{year}_{i}.tif"
            create_dummy_tiff(os.path.join(base_dir, dw_name), is_label=True)
            
            print(f"Created {s2_name} and {dw_name}")

    print("\nDone! You can now run the training script.")
    print(f"Try running: python model/train.py --data_dir {base_dir} --year 2019")

if __name__ == "__main__":
    main()
