import torch
import torch.nn as nn


class CustomResNeSt(nn.Module):
    def __init__(self, model_name='resnest50', pretrained=True):
        super().__init__()

        self.model = torch.hub.load('zhanghang1989/ResNeSt', model_name, pretrained=pretrained)
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        x = self.model(x)
        return x
