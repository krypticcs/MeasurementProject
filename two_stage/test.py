#imports
import os
import argparse
import matplotlib.pyplot as plt
import pandas as pd

import torch

from common.data import MeasurementSplit
from common.geometry import MEASUREMENT_NAMES
from two_stage.data_utils.data_utils import ShaftSet, load_stage1, load_segmenter, make_crops, positions_to_measurements
from two_stage.model.model import ShaftKeypointNet

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

parser = argparse.ArgumentParser(description = "Two-stage model testing")

parser.add_argument("--mdata", type = str, required = True, help = "Path to measurement data file (from ./data/measurements)")
parser.add_argument("--ds", type = str, required = True, help = "Dataset to test on (train, valid, test)")

parser.add_argument("--makecsv", action = "store_true", help = "Make a CSV of predicted vs actual measurements")

parser.add_argument("--path", type = str, default = "current_best.pth", required = False, help = "Stage-2 model path (from ./two_stage/model_saves/)")
parser.add_argument("--s1", type = str, default = "current_best.pth", required = False, help = "Stage-1 seg_heatmaps weights (from ./seg_heatmaps/model_saves/)")
parser.add_argument("--n_img", type = int, default = 30, required = False, help = "Number of crops to plot with predicted positions")
parser.add_argument("--seed", type = int, default = 42, required = False, help = "Torch seed")
parser.add_argument("--train_split", type = float, default = 0.8, required = False, help = "Training set split")
parser.add_argument("--val_split", type = float, default = 0.1, required = False, help = "Validation set split")
parser.add_argument("--test_bs", type = int, default = 64, required = False, help = "Testing batch size")

args = parser.parse_args()

test_set_name = args.ds.lower()
if test_set_name not in ["train", "valid", "test"]:
    raise ValueError(f"Unknown dataset name \"{test_set_name}\"")

print("\nSelected settings:\n")
print(f"Measurement file: {args.mdata}\nSelected dataset: {test_set_name}\nMake result CSV: {args.makecsv}\n")
print(f"Stage 1: ./seg_heatmaps/model_saves/{args.s1}\nStage 2: ./two_stage/model_saves/{args.path}\nTorch seed: {args.seed}\n")

#==========#

print("==========\n\nBegin dataset loading:\n")
split = MeasurementSplit(args.mdata, "box_images", (args.train_split, args.val_split), args.seed)
stage1 = load_stage1(args.s1, device)
segmenter = load_segmenter(device)
dataset = ShaftSet(split, test_set_name, stage1, segmenter, device)
del stage1, segmenter

model = ShaftKeypointNet(pretrained = False)
model.load_state_dict(torch.load(f"./two_stage/model_saves/{args.path}", map_location = "cpu"))
model.to(device)
model.eval()
print(f"\n==========\n\nModel loaded from ./two_stage/model_saves/{args.path}")

#testing: stage 2 replaces measurements 0-2, stage 1 keeps measurements 3-9
print("\n==========\n\nTesting started\n\n")
os.makedirs("./two_stage/test_results/img", exist_ok = True)

ypred = dataset.s1_meas.clone()
with torch.no_grad():
    for start in range(0, len(dataset), args.test_bs):
        idx = torch.arange(start, min(start + args.test_bs, len(dataset)))
        patches, centers, thetas, widths, _ = dataset.batch(idx, device)
        crops, s = make_crops(patches, centers, thetas, widths)
        pos, _ = model(crops)
        ypred[idx, :3] = positions_to_measurements(pos, s).cpu()

        for k in range(len(idx)):
            n = start + k
            if n >= args.n_img:
                break
            img_file, side = dataset.items[n]
            plt.figure(figsize = (6, 2))
            plt.imshow(crops[k, 0].cpu(), cmap = "gray", aspect = "equal")
            for p, label in zip(pos[k].tolist(), ["W", "X", "Y", "Z"]):
                plt.axvline(p, c = "r", lw = 0.8)
                plt.text(p + 1, 6, label, c = "r")
            plt.title(f"{img_file} side {side}  pred {ypred[n, :3].numpy().round(2)}  true {dataset.y[n, :3].numpy().round(2)}", fontsize = 6)
            plt.axis("off")
            plt.savefig(f"./two_stage/test_results/img/{n}.png", dpi = 150, bbox_inches = "tight")
            plt.close()

yvals = dataset.y
all_errs = torch.abs(ypred - yvals)
s1_errs = torch.abs(dataset.s1_meas - yvals)

if args.makecsv:
    resrows = []
    for i in range(len(dataset)):
        currow = {"id": i}
        for j, m in enumerate(MEASUREMENT_NAMES):
            currow[m + " (pred)"] = round(ypred[i, j].item(), 3)
            currow[m + " (true)"] = round(yvals[i, j].item(), 3)
            currow[m + " (err)"] = round((ypred[i, j] - yvals[i, j]).item(), 3)
        resrows.append(currow)
    finrow = {"id": "avg"}
    for j, m in enumerate(MEASUREMENT_NAMES):
        finrow[m + " (pred)"] = round(ypred[:, j].mean().item(), 3)
        finrow[m + " (true)"] = round(yvals[:, j].mean().item(), 3)
        finrow[m + " (err)"] = round(all_errs[:, j].mean().item(), 3)
    resrows.append(finrow)
    pd.DataFrame(resrows).to_csv("./two_stage/test_results/results.csv", index = False)

print("\n==========\n\nDone\n")

print("Percent error for each measurement (stage 1 only):")
print(" ".join(f"{v:.4f}" for v in (s1_errs / yvals).mean(dim = 0).tolist()))
print()

print("Percent error for each measurement:")
print(" ".join(f"{v:.4f}" for v in (all_errs / yvals).mean(dim = 0).tolist()))
print()

print("Mean of absolute errors:")
print([round(i, 3) for i in all_errs.mean(dim = 0).tolist()])
print()

print("Std dev of absolute errors:")
print([round(i, 3) for i in all_errs.std(dim = 0, unbiased = True).tolist()])
print()

print("Mean of true measurements:")
print([round(i, 3) for i in yvals.mean(dim = 0).tolist()])
print()

print("Std dev of true measurements:")
print([round(i, 3) for i in yvals.std(dim = 0, unbiased = True).tolist()])
print()
