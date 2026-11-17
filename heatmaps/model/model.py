#imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.geometry import heatmaps_to_keypoints

#heatmap model
class HeatmapsModel(nn.Module):
    def __init__(self, img_width, img_height, kp_cnt, channel_cnt = [1, 32, 64, 128, 256]):
        super().__init__()

        self.img_width = img_width
        self.img_height = img_height

        self.conv1 = nn.Sequential(
            nn.Conv2d(channel_cnt[0], channel_cnt[1], kernel_size = 7, stride = 1, padding = 3), #192, 240
            nn.BatchNorm2d(channel_cnt[1]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[1], channel_cnt[1], kernel_size = 3, stride = 1, padding = 1), #192, 240
            nn.BatchNorm2d(channel_cnt[1]),
            nn.ReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_cnt[1], channel_cnt[2], kernel_size = 3, stride = 2, padding = 1), #96, 120
            nn.BatchNorm2d(channel_cnt[2]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[2], channel_cnt[2], kernel_size = 3, stride = 1, padding = 1), #96, 120
            nn.BatchNorm2d(channel_cnt[2]),
            nn.ReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channel_cnt[2], channel_cnt[3], kernel_size = 3, stride = 2, padding = 1), #48, 60
            nn.BatchNorm2d(channel_cnt[3]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[3], channel_cnt[3], kernel_size = 3, stride = 1, padding = 1), #48, 60
            nn.BatchNorm2d(channel_cnt[3]),
            nn.ReLU(),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(channel_cnt[3], channel_cnt[4], kernel_size = 3, stride = 2, padding = 1), #24, 30
            nn.BatchNorm2d(channel_cnt[4]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[4], channel_cnt[4], kernel_size = 3, stride = 1, padding = 1), #24, 30
            nn.BatchNorm2d(channel_cnt[4]),
            nn.ReLU(),
        )

        self.conv5 = nn.Conv2d(channel_cnt[-1], kp_cnt, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = heatmaps_to_keypoints(x, self.img_height, self.img_width)
        return x
