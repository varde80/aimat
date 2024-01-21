import torchvision.models as models
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, block_type, layers, num_classes, input_size):
        super(ResNet, self).__init__()

        # Mapping block_type to the actual block class
        block_class = models.resnet.BasicBlock if block_type == 'BasicBlock' else models.resnet.Bottleneck

        # Creating the ResNet model
        if layers == [3, 4, 6, 3]:  # ResNet50
            self.model = models.resnet50(pretrained=False, num_classes=num_classes)
        elif layers == [3, 4, 23, 3]:  # ResNet101
            self.model = models.resnet101(pretrained=False, num_classes=num_classes)
        elif layers == [3, 8, 36, 3]:  # ResNet152
            self.model = models.resnet152(pretrained=False, num_classes=num_classes)
        else:
            # If none of the predefined architectures match, raise an error
            raise ValueError("Unsupported layers configuration for ResNet")

        # Adjusting the first convolution layer if input channels are not 3
        if input_size['channels'] != 3:
            self.model.conv1 = nn.Conv2d(input_size['channels'], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    def forward(self, x):
        return self.model(x)
    