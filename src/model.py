import torch
import torch.nn as nn
import torchvision.models as models
from timm import create_model


class TilesModel(nn.Module):
    def __init__(self, name, in_chans=1, num_classes=5, drop_prob=.0, pretrained=True, freeze=False, path=None):
        super(TilesModel, self).__init__()

        self.model = create_model(name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans, drop_rate=drop_prob)
       
        if path:
            self.load(path)

        if freeze:
            self.freeze()

    def freeze(self):
        for name, param in self.model.named_parameters():
            # if 'fc' not in name:
            param.requires_grad = False

        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                assert(param.requires_grad == False)

    def forward(self, x):
        x = self.model(x)
        return x

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))