import torch
import torch.nn as nn
import pretrainedmodels as pm
from torchvision import models


class ClotModelMIL(nn.Module):
    def __init__(self, num_crops=None):
        super().__init__()
        self.num_crops = num_crops

        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(base_model.children())[:-2])
        in_features_cnt = list(base_model.children())[-1].in_features
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features_cnt, 1),
            nn.Sigmoid(),
        )

    def freeze_encoder(self, flag):
        for param in self.model.parameters():
            param.requires_grad = not flag

    def forward(self, x):
        # x: bs x N x C x W x W
        bs, _, ch, w, h = x.shape
        x = x.view(bs * self.num_crops, ch, w, h)  # x: N bs x C x W x W
        x = self.model(x)  # x: N bs x C' x W' x W'

        # Concat and pool
        bs2, ch2, w2, h2 = x.shape
        x = x \
            .view(-1, self.num_crops, ch2, w2, h2) \
            .permute(0, 2, 1, 3, 4) \
            .contiguous() \
            .view(bs, ch2, self.num_crops * w2, h2)  # x: bs x C' x N W'' x W''
        return self.head(x)

    def save(self, model_path):
        weights = self.state_dict()
        torch.save(weights, model_path)

    def load(self, model_path):
        weights = torch.load(model_path, map_location='cpu')
        self.load_state_dict(weights)


class ClotModelSingle(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()

        if encoder_model == 'effnet_b0':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.model = base_model.features
            in_features_cnt = base_model.classifier[1].in_features
        elif encoder_model == 'resnet18':
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.model = nn.Sequential(*list(base_model.children())[:-2])
            in_features_cnt = list(base_model.children())[-1].in_features
        elif encoder_model == 'regnet_x_1_6gf':
            base_model = models.regnet_x_1_6gf(weights=models.RegNet_X_1_6GF_Weights.IMAGENET1K_V2)
            self.model = nn.Sequential(base_model.stem, base_model.trunk_output)
            in_features_cnt = base_model.fc.in_features
        else:
            raise Exception('Incorrect encoder name')

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features_cnt, 1),
            nn.Sigmoid(),
        )

    def freeze_encoder(self, flag):
        for param in self.model.parameters():
            param.requires_grad = not flag

    def forward(self, x):
        return self.head(self.model(x))

    def save(self, model_path):
        weights = self.state_dict()
        torch.save(weights, model_path)

    def load(self, model_path):
        weights = torch.load(model_path, map_location='cpu')
        self.load_state_dict(weights)
