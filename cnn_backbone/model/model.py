#imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from autoencoder.model.model import Autoencoder

#cnn on top of pretrained autoencoder encoder
class CNNModel(nn.Module):
    def __init__(self, img_width, img_height, ae_save_path = "./autoencoder/model_saves/autoencoder.pth", device = "cpu",  ae_has_resid = False, ae_ch_cnt = [1, 32, 64, 128, 256]):
        super().__init__()

        self.autoencoder = Autoencoder(ae_has_resid, ae_ch_cnt)
        self.autoencoder.load_state_dict(torch.load(ae_save_path, map_location = device))
        self.encoder = self.autoencoder.enc

        self.ae_fin_ch = ae_ch_cnt[-1]

        self.seq = nn.Sequential(
            nn.Conv2d(self.ae_fin_ch, 64, kernel_size = 3, padding = 1),
            nn.Conv2d(64, 256, kernel_size = 3, padding = 1),

            nn.Flatten(),
            nn.Linear(256 * 24 * 30, 4096),
            nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(4096, 11)
        )

    def forward(self, x):
        x = self.encoder(x)[-1]
        x = self.seq(x)
        return x
