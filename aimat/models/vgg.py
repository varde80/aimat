import torchvision.models as models
import torch.nn as nn

class VGG(nn.Module):
    def __init__(self, model_type, num_classes, input_size):
        super(VGG, self).__init__()

        # Creating the VGG model based on the model_type
        if model_type == 'vgg11':
            self.model = models.vgg11(pretrained=False, num_classes=num_classes)
        elif model_type == 'vgg13':
            self.model = models.vgg13(pretrained=False, num_classes=num_classes)
        elif model_type == 'vgg16':
            self.model = models.vgg16(pretrained=False, num_classes=num_classes)
        elif model_type == 'vgg19':
            self.model = models.vgg19(pretrained=False, num_classes=num_classes)
        else:
            # If an unsupported model_type is provided, raise an error
            raise ValueError("Unsupported model type for VGG")

        # Adjusting the first convolution layer if input channels are not 3
        if input_size['channels'] != 3:
            self.model.features[0] = nn.Conv2d(input_size['channels'], 64, kernel_size=3, padding=1)

    def forward(self, x):
        return self.model(x)
