import torch
import torch.nn as nn
import torchvision.models as models

def get_resnet18_cifar(num_classes=5, pretrained=False):
    """
    Returns a ResNet-18 model adapted for CIFAR-style 32x32 images.
    
    Modifications:
    1. Replaces the 7x7 conv (stride 2, padding 3) with a 3x3 conv (stride 1, padding 1) 
       to prevent immediate loss of spatial information on small 32x32 images.
    2. Replaces the MaxPool2d layer with an Identity layer, maintaining spatial size.
    3. Replaces the fc layer to output 'num_classes' instead of 1000.
    """
    if pretrained:
        # Load model with pre-trained weights on ImageNet
        # Note: Changing conv1 will re-initialize conv1 weights, but downstream blocks retain pre-trained features.
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)
        
    # 1. Modify the first conv layer
    model.conv1 = nn.Conv2d(
        in_channels=3, 
        out_channels=64, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        bias=False
    )
    
    # 2. Bypass the maxpool layer (replace with an Identity module)
    model.maxpool = nn.Identity()
    
    # 3. Replace the final linear classifier head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

if __name__ == "__main__":
    # Test model shape and output
    x = torch.randn(2, 3, 32, 32)
    model = get_resnet18_cifar(num_classes=5, pretrained=False)
    y = model(x)
    print("Input shape:", x.shape)
    print("Output shape (should be [2, 5]):", y.shape)
    
    # Check total parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params / 1e6:.2f}M")
