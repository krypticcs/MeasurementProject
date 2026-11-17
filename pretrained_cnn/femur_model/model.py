#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchxrayvision as xrv

#model
class FemurModel(nn.Module):
    def __init__(self, DBG = False):
        super().__init__()

        pretrained = xrv.models.DenseNet(weights="densenet121-res224-all")
        pre_feat = list(pretrained.features.children())

        self.backbone = nn.Sequential(*pre_feat[:7]) #512, 28, 28

        if DBG:
            for i, feat in enumerate(pre_feat):
                print(f'Feature {i}\n===================')
                print(feat)
                print()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), #256, 14, 14

            #nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.seq(x)
        return x
