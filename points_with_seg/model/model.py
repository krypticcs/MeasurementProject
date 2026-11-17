#imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from SEGMENTATIONS.mod_seg.model.model import SegUNet

#batchnorm relu class
class BNRelu(nn.Module):
    def __init__(self, chcnt, bn):
        super().__init__()
        
        if bn == "none":
            self.seq = nn.ReLU()
        elif bn == "before":
            self.seq = nn.Sequential(nn.BatchNorm2d(chcnt), nn.ReLU())
        elif bn == "after":
            self.seq = nn.Sequential(nn.ReLU(), nn.BatchNorm2d(chcnt))
        else:
            raise ValueError(f"Unknown batch norm setting: {bn}")

    def forward(self, x):
        x = self.seq(x)
        return x

#simple cnn pipeline (input is only the binary bone mask from the frozen pretrained seg model, not the raw image)
class SimpleCNNModel(nn.Module):
    def __init__(self, img_width, img_height, bn, seg_model_path):
        super().__init__()

        #frozen seg model
        self.seg_model = SegUNet(1)
        seg_weights = torch.load(seg_model_path, map_location = "cpu")
        self.seg_model.load_state_dict(seg_weights)
        for param in self.seg_model.parameters():
            param.requires_grad = False

        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            BNRelu(32, bn),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            BNRelu(64, bn),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            BNRelu(128, bn),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
            BNRelu(256, bn),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(256 * (img_width // 16) * (img_height // 16), 4096),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(4096, 11)
        )

    def forward(self, x):
        with torch.no_grad():
            seg_mask = (self.seg_model(x) > 0.5).float()
        x = seg_mask
        x = self.seq(x)
        return x
