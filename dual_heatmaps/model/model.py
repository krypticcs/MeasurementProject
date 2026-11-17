#imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from SEGMENTATIONS.mod_seg.model.model import Encoder

#dual encoder decoder (outputs raw heatmap logits for soft-argmax, so no final sigmoid)
class DualDecoder(nn.Module):
    def __init__(self, fin_ch = 1, channel_cnt = [1, 32, 64, 128, 256]):
        super().__init__()

        #bottleneck: 2*256 = 512
        self.conv1 = nn.Sequential(
            nn.Conv2d(2 * channel_cnt[4], channel_cnt[4], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[4]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[4], channel_cnt[3], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[3]),
            nn.ReLU(),
        )

        #after upsample + cat with 2*128: 128 + 256 = 384
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel_cnt[3] + 2 * channel_cnt[3], channel_cnt[3], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[3]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[3], channel_cnt[2], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[2]),
            nn.ReLU()
        )

        #after upsample + cat with 2*64: 64 + 128 = 192
        self.conv3 = nn.Sequential(
            nn.Conv2d(channel_cnt[2] + 2 * channel_cnt[2], channel_cnt[2], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[2]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[2], channel_cnt[1], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[1]),
            nn.ReLU()
        )

        #after upsample + cat with 2*32: 32 + 64 = 96
        self.conv4 = nn.Sequential(
            nn.Conv2d(channel_cnt[1] + 2 * channel_cnt[1], channel_cnt[1], kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(channel_cnt[1]),
            nn.ReLU(),
            nn.Conv2d(channel_cnt[1], fin_ch, kernel_size = 3, stride = 1, padding = 1)
        )

    def forward(self, seg_skips, raw_skips):
        x = torch.cat([seg_skips[4], raw_skips[4]], dim = 1)

        x = self.conv1(x)
        x = F.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners = False)
        x = torch.cat([x, seg_skips[3], raw_skips[3]], dim = 1)

        x = self.conv2(x)
        x = F.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners = False)
        x = torch.cat([x, seg_skips[2], raw_skips[2]], dim = 1)

        x = self.conv3(x)
        x = F.interpolate(x, scale_factor = 2, mode = 'bilinear', align_corners = False)
        x = torch.cat([x, seg_skips[1], raw_skips[1]], dim = 1)

        x = self.conv4(x)

        return x

#full dual encoder model
class DualSegHeatmaps(nn.Module):
    def __init__(self, fin_ch = 1, channel_cnt = [1, 32, 64, 128, 256]):
        super().__init__()

        self.seg_encoder = Encoder(channel_cnt)
        self.raw_encoder = Encoder(channel_cnt)
        self.decoder = DualDecoder(fin_ch, channel_cnt)

    def forward(self, x):
        seg_skips = self.seg_encoder(x)
        raw_skips = self.raw_encoder(x)
        x = self.decoder(seg_skips, raw_skips)
        return x
