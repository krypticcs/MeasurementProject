#imports
import os
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

#shared measurement data loader with on-the-fly augmentation
#every image is built from a cached copy at 2x the target resolution; train draws get fresh random augmentation,
#valid/test go through the same path with no augmentation, so train and eval images have the same resampling statistics

CACHE_ROOT = "./data/cache"

#####measurements + split#####

def read_measurements(measurement_file):
    df = pd.read_csv("./data/measurements/" + measurement_file)
    df = df.drop(columns = ["Student Name", "Notes", "Date Completed"] + [c for c in df.columns if c.startswith("Unnamed")], errors = "ignore")
    return df.dropna()

class MeasurementSplit:
    def __init__(self, measurement_file, img_dir, train_val_split, seed):
        self.img_dir = "./data/" + img_dir
        self.df = read_measurements(measurement_file)

        print(f"Used columns:\n{self.df.columns.values}\n")

        all_img_in_dir = [i for i in os.listdir(self.img_dir) if i.lower().endswith("dcm")]
        missing = sorted(set(self.df["ID"].values) - set(all_img_in_dir))
        if missing:
            raise SystemError(f"Directory is missing {len(missing)} images, e.g. {missing[:5]}")

        ids = set(self.df["ID"].values)
        all_img = [i for i in all_img_in_dir if i in ids]
        print(f"Images in use: {len(all_img)}/{len(all_img_in_dir)}\n")

        train_sz = int(train_val_split[0] * len(all_img))
        val_sz = int(train_val_split[1] * len(all_img))

        torch.manual_seed(seed)
        all_idx = torch.randperm(len(all_img))
        self.splits = {
            "train": [all_img[i] for i in all_idx[:train_sz]],
            "valid": [all_img[i] for i in all_idx[train_sz:train_sz + val_sz]],
            "test": [all_img[i] for i in all_idx[train_sz + val_sz:]],
        }

        self.y = {}
        for img_file, row in zip(self.df["ID"].values, self.df.drop(columns = "ID").values):
            self.y[img_file] = torch.from_numpy(row.astype(np.float32)).reshape(-1)

    #10 measurements for one side: unflipped image -> left set, flipped image -> right set
    def side_y(self, img_file, img_flip):
        y = self.y[img_file]
        return (y[10:] if img_flip else y[:10]).clone()

#####images#####

def load_dicom(path):
    img = torch.from_numpy(pydicom.dcmread(path).pixel_array.astype(np.float32))
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
    return img.unsqueeze(0).unsqueeze(0)

def resize(img, height, width):
    return F.interpolate(img, size = (height, width), mode = "bilinear", align_corners = False)

#per-image cache of the unflipped image at 2x the target size (float16), in RAM and on disk
class CachedImages:
    def __init__(self, img_dir, img_width, img_height):
        self.img_dir = img_dir
        self.disk_dir = os.path.join(CACHE_ROOT, f"{os.path.basename(os.path.normpath(img_dir))}_x2_{2 * img_width}x{2 * img_height}")
        self.cache_height = 2 * img_height
        self.cache_width = 2 * img_width
        os.makedirs(self.disk_dir, exist_ok = True)
        self.mem = {}

    def get(self, img_file):
        if img_file in self.mem:
            return self.mem[img_file]

        path = os.path.join(self.disk_dir, img_file + ".npy")
        if os.path.exists(path):
            x2 = torch.from_numpy(np.load(path))
        else:
            full = load_dicom(os.path.join(self.img_dir, img_file))
            x2 = resize(full, self.cache_height, self.cache_width).half()
            np.save(path, x2.numpy())

        self.mem[img_file] = x2
        return x2

#####augmentation helpers (same order and semantics as the original DataPoint)#####

#extra downscale so a rotated image still fits in the frame
def rot_fit_scale(rot, img_width, img_height):
    abscos = np.abs(np.cos(rot * np.pi / 180))
    abssin = np.abs(np.sin(rot * np.pi / 180))
    return min(1 / (abscos + img_height / img_width * abssin), 1 / (img_width / img_height * abssin + abscos))

#zero everything except the bottom-right (1 - aug_noise) region
def apply_mask(img, aug_noise):
    if aug_noise <= 0:
        return img
    _, _, height, width = img.shape
    out = torch.zeros_like(img)
    out[:, :, int(height * aug_noise):, int(width * aug_noise):] = img[:, :, int(height * aug_noise):, int(width * aug_noise):]
    return out

def pad_square(img, sq_dim):
    pad_sq = sq_dim - img.shape[2]
    pad_u = pad_sq // 2
    return F.pad(img, (0, 0, pad_u, pad_sq - pad_u), mode = "constant")

def uniform(rng):
    lo, hi = float(rng[0]), float(rng[1])
    return lo if hi <= lo else float(np.random.uniform(lo, hi))

def grid_values(rng):
    return list(np.arange(rng[0], rng[1] + 1e-8, rng[2]))

#####dataset#####

class AugDataset(Dataset):
    def __init__(self, split, cache, img_files, img_width, img_height, train, rot_rng, scale_rng, crop_rng, noise_rng, aug_repeats, sq_dim = None):
        self.split = split
        self.cache = cache
        self.img_width = img_width
        self.img_height = img_height
        self.train = train
        self.rot_rng = rot_rng
        self.scale_rng = scale_rng
        self.crop_rng = crop_rng
        self.noise_rng = noise_rng
        self.sq_dim = sq_dim

        for img_file in tqdm(img_files, unit = "image"):
            cache.get(img_file)

        sides = [(img_file, img_flip) for img_file in img_files for img_flip in range(2)]
        if train:
            #every draw gets fresh random augmentation; aug_repeats = draws per (image, side) per epoch
            augmented = any(float(r[1]) > float(r[0]) for r in [rot_rng, scale_rng, crop_rng, noise_rng])
            self.items = [(f, s, None) for f, s in sides] * (aug_repeats if augmented else 1)
        else:
            #never augmented; a noise grid (e.g. --mask in test scripts) adds deterministic masked copies
            self.items = [(f, s, n) for f, s in sides for n in grid_values(noise_rng)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_file, img_flip, aug_noise = self.items[idx]
        x2 = self.cache.get(img_file)
        y = self.split.side_y(img_file, img_flip)

        if self.train:
            aug_rot = uniform(self.rot_rng)
            aug_crop = uniform(self.crop_rng)
            aug_noise = uniform(self.noise_rng)
            aug_scale = uniform(self.scale_rng) * rot_fit_scale(aug_rot, self.img_width, self.img_height)
        else:
            aug_rot, aug_crop, aug_scale = 0, 1, 1

        img = x2.float()
        if img_flip:
            img = TF.hflip(img)
        img = apply_mask(img, aug_noise)

        _, _, height, width = img.shape
        img = img[:, :, int(height * (1 - aug_crop)):, int(width * (1 - aug_crop)):]

        new_height = int(self.img_height * aug_scale)
        new_width = int(self.img_width * aug_scale)
        img = resize(img, new_height, new_width)

        pad_height = self.img_height - new_height
        pad_width = self.img_width - new_width
        img = F.pad(img, (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2), mode = "constant")

        if aug_rot != 0:
            img = TF.rotate(img, angle = aug_rot, interpolation = TF.InterpolationMode.BILINEAR)
        if self.sq_dim is not None:
            img = pad_square(img, self.sq_dim)

        y[:9] *= aug_scale / aug_crop
        return img.squeeze(0), y, torch.tensor(float(aug_scale))

#####drop-in replacement for the per-folder DataProcessor#####

class DataProcessor:
    def __init__(self, measurement_file, img_dir, train_val_split, batch_size, img_width, img_height, seed, sq_dim = None):
        self.split = MeasurementSplit(measurement_file, img_dir, train_val_split, seed)
        self.cache = CachedImages(self.split.img_dir, img_width, img_height)
        self.batch_sizes = {"train": batch_size[0], "valid": batch_size[1], "test": batch_size[2]}
        self.img_width = img_width
        self.img_height = img_height
        self.sq_dim = sq_dim

    #ranges are (min, max, step); train samples uniformly in [min, max] (step ignored), valid/test are never augmented
    #except for the deterministic noise grid used to evaluate masked copies
    def create_ds(self, ds_name, rot_rng = (0, 0, 1), scale_rng = (1, 1, 1), crop_rng = (1, 1, 1), noise_rng = (0, 0, 1), aug_repeats = 4):
        if ds_name not in self.split.splits:
            raise ValueError(f"Unknown dataset name: {ds_name}")
        train = ds_name == "train"

        dataset = AugDataset(self.split, self.cache, self.split.splits[ds_name], self.img_width, self.img_height, train,
                             rot_rng, scale_rng, crop_rng, noise_rng, aug_repeats, self.sq_dim)
        print(f"Loaded {ds_name} with {len(dataset)} points{' (random augmentation each draw)' if train else ''}")

        dataloader = DataLoader(dataset, batch_size = self.batch_sizes[ds_name], shuffle = train, num_workers = 0)
        return dataset, dataloader
