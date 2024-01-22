import torchvision.models as models
import torch.nn as nn

class ResNeXt(nn.Module):
    def __init__(self, model_type, num_classes, input_size):
        super(ResNeXt, self).__init__()

        # Create the ResNeXt model based on the model_type
        if model_type == 'resnext50_32x4d':
            self.model = models.resnext50_32x4d(pretrained=False, num_classes=num_classes)
        elif model_type == 'resnext101_32x8d':
            self.model = models.resnext101_32x8d(pretrained=False, num_classes=num_classes)
        elif model_type == 'resnext101_64x4d':
            self.model = models.resnext101_64x4d(pretrained=False, num_classes=num_classes)
        else:
            # If an unsupported model_type is provided, raise an error
            raise ValueError("Unsupported model type for ResNeXt")

        # Adjusting the first convolution layer if input channels are not 3
        if input_size['channels'] != 3:
            self.model.conv1 = nn.Conv2d(input_size['channels'], 64, kernel_size=7, stride=2, padding=3, bias=False)

    def forward(self, x):
        return self.model(x)