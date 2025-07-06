import timm
import torch
import torch.nn as nn

from ..nn import normalize


class FeatureGenerator(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True, in_chans=in_channels)
        self.vit.head = nn.Identity()  # remove the classification head for timm version 0.6.12

    def forward(self, x):
        feat = self.vit.forward(x)  # for timm version 0.6.12
        return feat


class FeatureEmbedder(nn.Module):

    def __init__(self, in_channels: int = 768):
        super().__init__()
        self.bottleneck_layer_fc = nn.Linear(in_channels, 512)
        self.bottleneck_layer_fc.weight.data.normal_(0, 0.005)
        self.bottleneck_layer_fc.bias.data.fill_(0.1)
        self.bottleneck_layer = nn.Sequential(self.bottleneck_layer_fc, nn.ReLU(), nn.Dropout(0.5))

    def forward(self, x, norm_flag=True):
        feature = self.bottleneck_layer(x)
        if norm_flag:
            feature_norm = torch.norm(feature, p=2, dim=1, keepdim=True).clamp(min=1e-12)**0.5 * 2**0.5
            feature = torch.div(feature, feature_norm)
        return feature


class Classifier(nn.Module):

    def __init__(self, num_classes: int):
        super().__init__()
        self.classifier_layer = nn.Linear(512, num_classes)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, x, norm_flag=True):
        if (norm_flag):
            self.classifier_layer.weight.data = normalize(self.classifier_layer.weight, 2, axis=0)
        classifier_out = self.classifier_layer(x)
        return classifier_out
