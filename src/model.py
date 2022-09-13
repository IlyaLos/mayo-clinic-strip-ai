import torch
import torch.nn as nn
import pretrainedmodels as pm
from torchvision import models


class ClotModel(nn.Module):
    def __init__(self, encoder_model):
        super().__init__()

        if encoder_model == 'resnext50_32':
            self.model = pm.se_resnext50_32x4d(pretrained=None)
            in_features_cnt = self.model.last_linear.in_features
            self.model.last_linear = nn.Linear(in_features=in_features_cnt, out_features=1, bias=True)
            #self.model.layer0.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        elif encoder_model == 'effnet_b0':
            self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            in_features_cnt = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features=in_features_cnt, out_features=1, bias=True)
            #self.model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        else:
            raise Exception(f'Model name {encoder_model} is unknown. Use any of [resnext50_32, effnet_b0]')

        self.activation = nn.Sigmoid()

    def freeze_encoder(self, flag):
        for param in self.model.features.parameters():
            param.requires_grad = not flag

    def forward(self, x):
        return self.activation(self.model(x))

    def save(self, model_path):
        weights = self.state_dict()
        torch.save(weights, model_path)

    def load(self, model_path):
        weights = torch.load(model_path, map_location='cpu')
        self.load_state_dict(weights)
