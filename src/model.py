import torch
import torch.nn as nn
from torchvision import models


class ClotModelSingle(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()

        if encoder_model == 'effnet_b0':
            base_model = models.efficientnet_b0(pretrained=True)
            self.model = base_model.features
            in_features_cnt = base_model.classifier[1].in_features
        elif encoder_model == 'resnet18':
            base_model = models.resnet18(pretrained=True)
            self.model = nn.Sequential(*list(base_model.children())[:-2])
            in_features_cnt = list(base_model.children())[-1].in_features
        elif encoder_model == 'regnet_x_1_6gf':
            base_model = models.regnet_x_1_6gf(pretrained=True)
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
