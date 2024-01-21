import timm
import torch.nn as nn
from torch.nn import functional as F

class ViT(nn.Module):
    def __init__(self, model_type, pretrained, num_classes):
        super(ViT, self).__init__()

        # Create the ViT model based on the model_type
        self.model = timm.create_model(model_type, 
                                       pretrained=pretrained, 
                                       num_classes=num_classes, 
                                       )

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        assert x.size(-1) == 224 and x.size(-2) == 224, "Input size must be 224x224 for ViT models, got {}x{}".format(x.size(-2), x.size(-1))
        return self.model(x)
