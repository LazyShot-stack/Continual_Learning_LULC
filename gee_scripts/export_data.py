import ee
import os

# Initialize GEE
# IMPORTANT: You must replace 'your-project-id' with your actual Google Cloud Project ID for Earth Engine.
# If you don't have one, go to https://code.earthengine.google.com/register
PROJECT_ID = 'formidable-code-280411' 

try:
    ee.Initialize(project=PROJECT_ID)
except Exception as e:
    print("Authenticating GEE...")
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

def get_sentinel2_composite(roi, year):
    """
    Generates a cloud-free median composite for Sentinel-2 for a given year.
    """
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    
    # Simple cloud masking (can be improved)
    def mask_s2_clouds(image):
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0) \
            .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask).divide(10000)

    composite = s2.map(mask_s2_clouds).median().clip(roi)
    
    # Select bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
    return composite.select(['B2', 'B3', 'B4', 'B8'])

def get_dynamic_world_labels(roi, year):
    """
    Fetches Dynamic World LULC labels (mode) for a given year.
    """
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'
    
    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date)
    
    # Get the mode (most frequent class) for the year
    classification = dw.select('label').mode().clip(roi)
    return classification

def export_image(image, description, folder, scale=10, region=None):
    """
    Exports an image to Google Drive.
    """
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        scale=scale,
        region=region,
        fileFormat='GeoTIFF',
        maxPixels=1e13
    )
    task.start()
    print(f"Started export task: {description}")

def main():
    # Define ROI (Example: A bounding box around a city, e.g., Bangalore)
    # You can change this to your specific area of interest
    roi = ee.Geometry.Rectangle([77.5, 12.9, 77.7, 13.1]) 
    
    years = range(2018, 2026) # 2018 to 2025
    
    output_folder = "LULC_Continual_Learning_Data"
    
    for year in years:
        print(f"Processing year: {year}")
        
        # 1. Sentinel-2 Composite
        s2_img = get_sentinel2_composite(roi, year)
        export_image(s2_img, f"Sentinel2_{year}", output_folder, scale=10, region=roi)
        
        # 2. Dynamic World Labels
        dw_img = get_dynamic_world_labels(roi, year)
        export_image(dw_img, f"DynamicWorld_{year}", output_folder, scale=10, region=roi)

if __name__ == "__main__":
    main()
