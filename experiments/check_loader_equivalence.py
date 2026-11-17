#run from project root: python -m experiments.check_loader_equivalence [--n 40]
#compares common/data.py against the original per-folder DataPoint/DataProcessor (copied below verbatim as the reference):
#  1. same train/valid/test split
#  2. valid/test (never augmented): same labels, images equal within resize tolerance (new path resizes from a 2x cache)
#  3. train with forced augmentation parameters: same labels/scale, images close to the old DataPoint with those parameters
import os
import argparse
import numpy as np
import pandas as pd
import pydicom

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import common.data as data

parser = argparse.ArgumentParser(description = "Check the shared loader against the original DataPoint")
parser.add_argument("--mdata", type = str, default = "20260427.csv", required = False)
parser.add_argument("--idata", type = str, default = "box_images", required = False)
parser.add_argument("--n", type = int, default = 40, required = False, help = "Images per check")
args = parser.parse_args()

#####reference: original DataProcessor split and DataPoint transform#####

def legacy_split(measurement_file, img_dir, train_val_split, seed):
    df = pd.read_csv("./data/measurements/" + measurement_file)
    df.drop(columns = ["Student Name", "Notes", "Date Completed", "Unnamed: 24", "Unnamed: 25", "Unnamed: 26", "Unnamed: 27", "Unnamed: 28", "Unnamed: 29", "Unnamed: 30", "Unnamed: 31"], inplace = True, errors = "ignore")
    df = df.dropna()
    all_img_in_dir = [i for i in os.listdir(img_dir) if i.lower().endswith("dcm")]
    all_img = [i for i in all_img_in_dir if i in df["ID"].values]
    train_sz = int(train_val_split[0] * len(all_img))
    val_sz = int(train_val_split[1] * len(all_img))
    torch.manual_seed(seed)
    all_idx = torch.randperm(len(all_img))
    return df, {"train": [all_img[i] for i in all_idx[:train_sz]], "valid": [all_img[i] for i in all_idx[train_sz:train_sz + val_sz]], "test": [all_img[i] for i in all_idx[train_sz + val_sz:]]}

def legacy_point(img_path, img_file, img_flip, df, img_width, img_height, aug_rot = 0, aug_scale = 1, aug_crop = 1, aug_noise = 0, sq_dim = None):
    y = torch.from_numpy(df.loc[df["ID"] == img_file].drop(columns = "ID").values.astype(np.float32)).reshape(-1)
    abscos = np.abs(np.cos(aug_rot * np.pi / 180))
    abssin = np.abs(np.sin(aug_rot * np.pi / 180))
    aug_scale *= min(1 / (abscos + img_height / img_width * abssin), 1 / (img_width / img_height * abssin + abscos))

    img = torch.from_numpy(pydicom.dcmread(img_path).pixel_array).float()
    img = (img - img.min()) / (img.max() - img.min() + 1e-5)
    img = img.unsqueeze(0).unsqueeze(0)
    if img_flip:
        img = TF.hflip(img)

    _, _, full_height, full_width = img.shape
    msk = torch.ones_like(img)
    msk[:, :, int(full_height * aug_noise):, int(full_width * aug_noise):] = 0
    img = img * (1 - msk)

    _, _, orig_height, orig_width = img.shape
    img = img[:, :, int(orig_height * (1 - aug_crop)):, int(orig_width * (1 - aug_crop)):]
    new_height = int(img_height * aug_scale)
    new_width = int(img_width * aug_scale)
    img = F.interpolate(img, size = (new_height, new_width), mode = "bilinear", align_corners = False)
    pad_height = img_height - new_height
    pad_width = img_width - new_width
    img = F.pad(img, (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2), mode = "constant")
    img = TF.rotate(img, angle = aug_rot, interpolation = TF.InterpolationMode.BILINEAR)
    if sq_dim is not None:
        pad_sq = sq_dim - img_height
        img = F.pad(img, (0, 0, pad_sq // 2, pad_sq - pad_sq // 2), mode = "constant")

    y_aug = (y[10:] if img_flip else y[:10]).clone()
    y_aug[:9] *= aug_scale / aug_crop
    return img.squeeze(0), y_aug, torch.tensor(aug_scale).float()

#####checks#####

img_dir = "./data/" + args.idata
df, old_splits = legacy_split(args.mdata, img_dir, (0.8, 0.1), 42)

configs = [("240x192 (cnn_on_points, cnn_backbone, points_with_seg, heatmaps, seg_heatmaps, dual_heatmaps)", 240, 192, None),
           ("224x179 + square pad 224 (pretrained_cnn)", 224, 179, 224)]

sample_uniform = data.uniform
ok = True
for cfg_name, width, height, sq_dim in configs:
    print(f"\n===== {cfg_name} =====")
    dp = data.DataProcessor(args.mdata, args.idata, (0.8, 0.1), (8, 8, 8), width, height, 42, sq_dim = sq_dim)

    same_split = all(dp.split.splits[k] == old_splits[k] for k in old_splits)
    print(f"split identical: {same_split}  sizes {[len(dp.split.splits[k]) for k in ['train', 'valid', 'test']]}")
    ok &= same_split

    for ds_name in ["valid", "test"]:
        files = dp.split.splits[ds_name][:args.n // 2]
        ds = data.AugDataset(dp.split, dp.cache, files, width, height, False, (0, 0, 1), (1, 1, 1), (1, 1, 1), (0, 0, 1), 1, sq_dim)
        diffs, label_ok = [], True
        for i, (f, side, _) in enumerate(ds.items):
            new_img, new_y, new_s = ds[i]
            old_img, old_y, old_s = legacy_point(f"{img_dir}/{f}", f, side, df, width, height, sq_dim = sq_dim)
            label_ok &= torch.allclose(new_y, old_y) and torch.allclose(new_s, old_s)
            diffs.append((new_img - old_img).abs())
        d = torch.stack(diffs)
        print(f"{ds_name}: {len(ds)} points, labels identical: {label_ok}, |pixel diff| mean {d.mean():.4f} max {d.max():.4f} (intensities in [0, 1])")
        ok &= label_ok and d.mean().item() < 0.02

    #forced augmentation: patch the sampler so the train path uses exactly these parameters
    files = dp.split.splits["train"][:args.n // 4]
    combos = [(10, 0.75, 0.8, 0.25), (-10, 1.0, 0.6, 0.0), (5, 0.5, 1.0, 0.5)]
    for rot, scale, crop, noise in combos:
        vals = iter([rot, crop, noise, scale] * (2 * len(files)))
        data.uniform = lambda rng: next(vals)
        ds = data.AugDataset(dp.split, dp.cache, files, width, height, True, (-10, 10, 10), (0.5, 1, 0.25), (0.6, 1, 0.2), (0, 0.5, 0.25), 1, sq_dim)
        diffs, label_ok = [], True
        for i, (f, side, _) in enumerate(ds.items):
            new_img, new_y, new_s = ds[i]
            old_img, old_y, old_s = legacy_point(f"{img_dir}/{f}", f, side, df, width, height, rot, scale, crop, noise, sq_dim)
            label_ok &= torch.allclose(new_y, old_y, rtol = 1e-5) and torch.allclose(new_s, old_s)
            diffs.append((new_img - old_img).abs())
        d = torch.stack(diffs)
        print(f"train rot {rot} scale {scale} crop {crop} mask {noise}: labels identical: {label_ok}, |pixel diff| mean {d.mean():.4f}")
        ok &= label_ok and d.mean().item() < 0.03
    data.uniform = sample_uniform

print(f"\nALL CHECKS PASSED: {ok}")
