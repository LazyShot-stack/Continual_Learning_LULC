import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class UrbanMapper(nn.Module):
    def __init__(self, num_classes, num_channels=4):
        """
        DeepLabv3+ model with ResNet-50 backbone.
        Modified to accept 'num_channels' input (default 4 for Sentinel-2: B, G, R, NIR).
        """
        super(UrbanMapper, self).__init__()
        
        # ===== IMPROVEMENT: Load pre-trained DeepLabV3 with aux_loss for better training =====
        # Note: Pre-trained weights have 21 classes (COCO), so load without num_classes first
        self.model = deeplabv3_resnet50(pretrained=True, aux_loss=True)
        
        # Modify the first layer to accept 'num_channels' instead of 3
        # We average the weights of the first layer to initialize the new channels if num_channels > 3
        original_first_layer = self.model.backbone.conv1
        new_first_layer = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        with torch.no_grad():
            if num_channels > 3:
                # Copy weights for the first 3 channels
                new_first_layer.weight[:, :3] = original_first_layer.weight
                # Initialize extra channels with average of RGB weights
                avg_weight = torch.mean(original_first_layer.weight, dim=1, keepdim=True)
                new_first_layer.weight[:, 3:] = avg_weight.repeat(1, num_channels - 3, 1, 1)
            else:
                new_first_layer.weight[:, :num_channels] = original_first_layer.weight[:, :num_channels]
                
        self.model.backbone.conv1 = new_first_layer
        
        # Modify the classifier to output 'num_classes'
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        if hasattr(self.model, 'aux_classifier'):
            self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        output = self.model(x)
        # Handle aux_loss output format (dict with 'out' and 'aux')
        if isinstance(output, dict):
            return output['out']
        return output

if __name__ == "__main__":
    # Test the model
    model = deeplabv3_resnet50(pretrained=True, num_classes=9)
    x = torch.randn(2, 4, 256, 256)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
