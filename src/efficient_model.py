import sys

import torch
import torchvision
from torch import nn
from torchvision import models
from pretrainedmodels import se_resnext101_32x4d, se_resnext50_32x4d, senet154
from pretrainedmodels import inceptionresnetv2
from efficientnet_pytorch import EfficientNet
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter



class Mish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


class EfficientHead(nn.Module):
    def __init__(self, n_in_features):
        super(EfficientHead, self).__init__()

        self.cnn_head = nn.Sequential(
            Mish(), nn.Conv2d(n_in_features, 512, kernel_size=3),
            nn.BatchNorm2d(512), GeM(), Flatten())

        self.fc = nn.Linear(512, 5)

    def forward(self, x):

        x = self.cnn_head(x)
        out = self.fc(x)
        return out


class EfficientHeadV2(nn.Module):
    def __init__(self, n_in_features, num_classes):
        super(EfficientHeadV2, self).__init__()

        self.cnn_head = nn.Sequential(
            Mish(), nn.Conv2d(n_in_features, 512, kernel_size=3),
            nn.BatchNorm2d(512), GeM(), Flatten())

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):

        x = self.cnn_head(x)
        out = self.fc(x)
        return out


class Efficient(nn.Module):
    def __init__(self, num_classes, encoder='efficientnet-b0', pool_type="avg"):
        super().__init__()
        n_channels_dict = {'efficientnet-b0': 1280, 'efficientnet-b1': 1280, 'efficientnet-b2': 1408,
                           'efficientnet-b3': 1536, 'efficientnet-b4': 1792, 'efficientnet-b5': 2048,
                           'efficientnet-b6': 2304, 'efficientnet-b7': 2560}
        self.net = EfficientNet.from_pretrained(encoder)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if pool_type == "concat":
            self.net.avg_pool = AdaptiveConcatPool2d()
            out_shape = n_channels_dict[encoder]*2
        elif pool_type == "avg":
            self.net.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            out_shape = n_channels_dict[encoder]
        elif pool_type == "gem":
            self.net.avg_pool = GeM()
            out_shape = n_channels_dict[encoder]
        # self.classifier = EfficientHead(out_shape) 
        self.classifier = EfficientHeadV2(out_shape, num_classes)


    def forward(self, x):

        # x = x.repeat(1, 3, 1, 1)
        x = self.net.extract_features(x)
        # x = self.net.avg_pool(x)
        x = self.classifier(x)
        return x
