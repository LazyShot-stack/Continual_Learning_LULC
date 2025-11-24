# quick-debug_shapes.py
from model import SentinelDataset  # adapt import to your dataset
ds = SentinelDataset(data_root, data_root)        # instantiate same way as train script
for i in range(20):
    sample = ds[i]
    # if sample is (img, mask) tuple:
    if isinstance(sample, tuple) or isinstance(sample, list):
        img, mask = sample[0], sample[1]
    else:
        # if dataset returns dict
        img = sample.get('image') if isinstance(sample, dict) else None
        mask = sample.get('mask') if isinstance(sample, dict) else None
    print(i, getattr(img, 'shape', None), getattr(mask, 'shape', None))
