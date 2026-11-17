#imports
import torch
import torch.nn as nn
import torch.nn.functional as F

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

#simple cnn pipeline
class SimpleCNNModel(nn.Module):
    def __init__(self, img_width, img_height, bn):
        super().__init__()
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
        x = self.seq(x)
        return x

#alexnet pipeline
class AlexNet(nn.Module):
    def __init__(self, img_width, img_height, bn):
        super().__init__()

        conv_sz = lambda dim, ksz, str, pd : (dim + 2 * pd - ksz) // str + 1
        pool_sz = lambda dim, ksz, str : (dim - ksz) // str + 1

        fin_w = img_width
        fin_h = img_height

        fin_w = pool_sz(conv_sz(fin_w, 11, 4, 2), 3, 2)
        fin_h = pool_sz(conv_sz(fin_h, 11, 4, 2), 3, 2)

        fin_w = pool_sz(conv_sz(fin_w, 5, 1, 2), 3, 2)
        fin_h = pool_sz(conv_sz(fin_h, 5, 1, 2), 3, 2)

        fin_w = conv_sz(fin_w, 3, 1, 1)
        fin_h = conv_sz(fin_h, 3, 1, 1)

        fin_w = conv_sz(fin_w, 3, 1, 1)
        fin_h = conv_sz(fin_h, 3, 1, 1)

        fin_w = pool_sz(conv_sz(fin_w, 3, 1, 1), 3, 2)
        fin_h = pool_sz(conv_sz(fin_h, 3, 1, 1), 3, 2)

        self.seq = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size = 11, stride = 4, padding = 2),
            BNRelu(96, bn),
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2),
            BNRelu(256, bn),
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1),
            BNRelu(384, bn),

            nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1),
            BNRelu(384, bn),

            nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1),
            BNRelu(256, bn),
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            nn.Flatten(),
            #nn.Dropout(),
            nn.Linear(256 * fin_w * fin_h, 4096),
            nn.ReLU(),
            #nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 11)

        )

    def forward(self, x):
        x = self.seq(x)
        return x
