#imports
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from seg_heatmaps.data_utils.data_utils import DataProcessor
from SEGMENTATIONS.mod_seg.model.model import SegUNet
from common.geometry import PIX_PER_MM, HEATMAP_LABELS, HEATMAP_ABS_LABELS, MEASUREMENT_NAMES, measurements_to_coord_heatmap, coord_to_measurements, abs_kp_to_coord, get_ab, heatmaps_to_keypoints

#==========#

#device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

#parser

parser = argparse.ArgumentParser(description = "Model testing")

parser.add_argument("--mdata", type = str, required = True, help = "Path to measurement data file (from ./data/measurements)")
parser.add_argument("--ds", type = str, required = True, help = "Dataset to test on (train, valid, test)")

parser.add_argument("--makecsv", action = "store_true", help = "Make a CSV of predicted vs actual measurements")
parser.add_argument("--mask", action = "store_true", help = "Also evaluate masked copies of each image (same masking levels as training)")

parser.add_argument("--path", type = str, default = "current_best.pth", required = False, help = "Model path (from ./seg_heatmaps/model_saves/)")
parser.add_argument("--seed", type = int, default = 42, required = False, help = "Torch seed")
parser.add_argument("--train_split", type = float, default = 0.8, required = False, help = "Training set split")
parser.add_argument("--val_split", type = float, default = 0.1, required = False, help = "Validation set split")
parser.add_argument("--train_bs", type = int, default = 1, required = False, help = "Training set batch size")
parser.add_argument("--val_bs", type = int, default = 1, required = False, help = "Validation set batch size")
parser.add_argument("--test_bs", type = int, default = 1, required = False, help = "Testing set batch size")

args = parser.parse_args()

#settings

pix_per_mm = PIX_PER_MM
img_scale_factor = 0.1
img_width = 240
img_height = 192

measurement_file = args.mdata
test_set_name = args.ds.lower()

make_csv = args.makecsv
use_mask = args.mask

model_path = args.path
seed = args.seed
train_split = args.train_split
val_split = args.val_split
train_batch_size = args.train_bs
val_batch_size = args.val_bs
test_batch_size = args.test_bs
batch_sizes = (train_batch_size, val_batch_size, test_batch_size)

#checks
if test_set_name not in ["train", "valid", "test"]:
    raise ValueError(f"Unknown dataset name \"{test_set_name}\"")

print("\nSelected settings:\n")
print(f"Measurement file: {measurement_file}\nSelected dataset: {test_set_name}\nMake result CSV: {make_csv}\nMasked copies: {use_mask}\n")
print(f"Torch seed: {seed}\n")
print(f"Data split: {(train_split, val_split, 1 - train_split - val_split)}")
print(f"Batch size: {batch_sizes}\n")

#==========#

print("==========\n\nBegin dataset loading:\n")
NOISE_AUG = (0, 0.5, 0.25) if use_mask else (0, 0, 1)
data_processor = DataProcessor(measurement_file, (train_split, val_split), batch_sizes, img_width, img_height, seed)
dataset, dataloader = data_processor.create_ds(test_set_name, noise_rng = NOISE_AUG)

#==========#

#model set up
model = SegUNet(13, final_act = False)

model.load_state_dict(torch.load(f"./seg_heatmaps/model_saves/{model_path}", map_location = "cpu"))
print(f"\n==========\n\nModel loaded from ./seg_heatmaps/model_saves/{model_path}")

#testing
print("\n==========\n\nTesting started\n\n")

model.to(device)
model.eval()

all_errs = None
total_percent_err = torch.zeros(10).to(device)
all_measurements = None


mnames = MEASUREMENT_NAMES
mcols = []
for i in mnames:
    mcols.append(i + " (pred)")
    mcols.append(i + " (true)")
    mcols.append(i + " (err)")
resrows = []
finrow = {"id": "avg"}
for i in mcols:
    finrow[i] = 0

os.makedirs("./seg_heatmaps/test_results/img", exist_ok=True)
with torch.no_grad():
    for idx, (images, yvals, aug_scales) in enumerate(tqdm(dataloader, unit = "batch")):
        images = images.to(device)
        yvals = yvals.to(device)
        aug_scales = aug_scales.to(device)

        hm = model(images)
        abs_coord = heatmaps_to_keypoints(hm)
        model_coord = abs_kp_to_coord(abs_coord)
        ab = get_ab(model_coord)
        real_coord = measurements_to_coord_heatmap(yvals, ab, pix_per_mm, img_scale_factor)
        ypred = coord_to_measurements(model_coord, pix_per_mm, img_scale_factor, flip_y = True)

        errs = torch.abs(yvals - ypred)
        if all_errs is None:
            all_errs = errs
            all_measurements = yvals
        else:
            all_errs = torch.cat((all_errs, errs), dim = 0)
            all_measurements = torch.cat((all_measurements, yvals), dim = 0)
        total_percent_err += torch.sum(torch.abs(yvals - ypred) / yvals, dim = 0)

        plt.figure(figsize = (6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(images[0][0].detach().cpu(), cmap = 'gray')
        plt.scatter(abs_coord[0, :, 0].detach().cpu(), abs_coord[0, :, 1].detach().cpu(), s = 10)
        for i, label in enumerate(HEATMAP_ABS_LABELS):
            curx, cury = abs_coord[0, i]
            curx = curx.detach().cpu()
            cury = cury.detach().cpu()
            plt.text(curx + 0.5, cury + 0.5, label, c = 'r')

        plt.subplot(1, 2, 2)
        plt.scatter(model_coord[0, :, 0].detach().cpu(), model_coord[0, :, 1].detach().cpu(), c = 'r')
        for i, label in enumerate(HEATMAP_LABELS):
            curx, cury = model_coord[0, i]
            curx = curx.detach().cpu()
            cury = cury.detach().cpu()
            plt.text(curx + 0.5, cury + 0.5, label)
        plt.scatter(real_coord[0, :, 0].detach().cpu(), real_coord[0, :, 1].detach().cpu(), c = 'b')
        for i, label in enumerate(HEATMAP_LABELS):
            curx, cury = real_coord[0, i]
            curx = curx.detach().cpu()
            cury = cury.detach().cpu()
            plt.text(curx + 0.5, cury + 0.5, label)
        plt.gca().invert_yaxis()
        plt.xlabel('x (pixels)')
        plt.ylabel('y (pixels)')

        plt.savefig(f"./seg_heatmaps/test_results/img/{idx}.png")
        plt.close()

        if make_csv:
            for i in range(yvals.shape[0]):
                currow = {"id": idx * yvals.shape[0] + i}
                for j in range(10):
                    v1 = ypred[i][j].item()
                    v2 = yvals[i][j].item()
                    v3 = v1 - v2

                    finrow[mcols[3 * j]] += v1
                    finrow[mcols[3 * j + 1]] += v2
                    finrow[mcols[3 * j + 2]] += abs(v3)

                    currow[mcols[3 * j]] = round(v1, 3)
                    currow[mcols[3 * j + 1]] = round(v2, 3)
                    currow[mcols[3 * j + 2]] = round(v3, 3)
                resrows.append(currow)

if make_csv:
    for i in mcols:
        finrow[i] = round(finrow[i] / len(dataset), 3)
    resrows.append(finrow)
    df = pd.DataFrame(resrows)
    df.to_csv("./seg_heatmaps/test_results/results.csv", index = False)

print("\n==========\n\nDone\n")

print(f'Percent error for each measurement:')
for i in total_percent_err:
    print(f'{(i.item() / len(dataset)):.4f}', end = ' ')
print()

print("Mean of absolute errors:")
print([round(i, 3) for i in all_errs.mean(dim = 0).tolist()])
print()

print("Std dev of absolute errors:")
print([round(i, 3) for i in all_errs.std(dim = 0, unbiased = True).tolist()])
print()

print("Mean of true measurements:")
print([round(i, 3) for i in all_measurements.mean(dim = 0).tolist()])
print()

print("Std dev of true measurements:")
print([round(i, 3) for i in all_measurements.std(dim = 0, unbiased = True).tolist()])
print()
