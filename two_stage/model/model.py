#imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

def conv1d_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size = 5, padding = 2),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(),
        nn.Conv1d(out_ch, out_ch, kernel_size = 5, padding = 2),
        nn.BatchNorm1d(out_ch),
        nn.ReLU()
    )

#1D soft-argmax over the last dim; returns positions in input-column units of a profile that was downsampled by `stride`
def soft_argmax_1d(logits, stride):
    L = logits.shape[-1]
    prob = F.softmax(logits, dim = -1)
    pos = (torch.arange(L, device = logits.device, dtype = logits.dtype) + 0.5) * stride - 0.5
    return (prob * pos).sum(dim = -1)

#stage 2: shaft crop (B, 1, H, W) -> x-positions (in crop columns) of W, X, Y, Z
#2D ResNet-18 stem (ImageNet weights) -> features mean-pooled over height -> 1D U-Net decoder -> 4 1D heatmaps -> soft-argmax
class ShaftKeypointNet(nn.Module):
    def __init__(self, pretrained = True, n_points = 4):
        super().__init__()
        backbone = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu) #64ch, /2
        self.pool = backbone.maxpool
        self.layer1 = backbone.layer1 #64ch, /4
        self.layer2 = backbone.layer2 #128ch, /8

        self.dec3 = conv1d_block(128, 128)
        self.dec2 = conv1d_block(128 + 64, 64)
        self.dec1 = conv1d_block(64 + 64, 64)
        self.head = nn.Conv1d(64, n_points, kernel_size = 1)

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)
        f1 = self.stem(x)
        f2 = self.layer1(self.pool(f1))
        f3 = self.layer2(f2)

        p1, p2, p3 = f1.mean(dim = 2), f2.mean(dim = 2), f3.mean(dim = 2)

        d = self.dec3(p3)
        d = F.interpolate(d, size = p2.shape[-1], mode = "linear", align_corners = False)
        d = self.dec2(torch.cat((d, p2), dim = 1))
        d = F.interpolate(d, size = p1.shape[-1], mode = "linear", align_corners = False)
        d = self.dec1(torch.cat((d, p1), dim = 1))

        logits = self.head(d)
        return soft_argmax_1d(logits, x.shape[-1] / logits.shape[-1]), logits
