#imports
import os
import argparse
from tqdm import tqdm

import torch
import torch.optim as optim

from common.data import MeasurementSplit
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

parser = argparse.ArgumentParser(description = "Two-stage model training (stage 2 on shaft crops, stage 1 frozen)")

parser.add_argument("--mdata", type = str, required = True, help = "Path to measurement data file (from ./data/measurements)")
parser.add_argument("--s1", type = str, default = "current_best.pth", required = False, help = "Frozen stage-1 seg_heatmaps weights (from ./seg_heatmaps/model_saves/)")
parser.add_argument("--scale", type = float, default = 0.1, required = False, help = "Stage-1 image scale (must match the seg_heatmaps checkpoint)")

parser.add_argument("--epoch", type = int, default = 100, required = False, help = "Max number of epochs")
parser.add_argument("--lr", type = float, default = 5e-4, required = False, help = "Learning rate")
parser.add_argument("--esl", type = int, default = 10, required = False, help = "Number of epochs with no improvement to early stop training")
parser.add_argument("--repeats", type = int, default = 4, required = False, help = "Jittered crops per (image, side) per epoch")
parser.add_argument("--run", type = str, default = "", required = False, help = "Run name: saves <run>_best.pth / <run>_final.pth instead of current_best.pth / final_model.pth")
parser.add_argument("--load", type = str, default = "", required = False, help = "Load a stage-2 model to continue training (./two_stage/model_saves)")
parser.add_argument("--seed", type = int, default = 42, required = False, help = "Torch seed")
parser.add_argument("--train_split", type = float, default = 0.8, required = False, help = "Training set split")
parser.add_argument("--val_split", type = float, default = 0.1, required = False, help = "Validation set split")
parser.add_argument("--train_bs", type = int, default = 64, required = False, help = "Training set batch size")
parser.add_argument("--val_bs", type = int, default = 64, required = False, help = "Validation set batch size")

args = parser.parse_args()

best_name = f"{args.run}_best.pth" if args.run else "current_best.pth"
final_name = f"{args.run}_final.pth" if args.run else "final_model.pth"
save_dir = "./two_stage/model_saves"

print("\nSelected settings:\n")
print(f"Measurement file: {args.mdata}\nStage 1: ./seg_heatmaps/model_saves/{args.s1}\nStage-1 scale: {args.scale}\n")
print(f"Max epoch cnt: {args.epoch}\nLearning rate: {args.lr}\nEarly stopping limit: {args.esl}\nRepeats: {args.repeats}\nLoad pretrained: {args.load}\nTorch seed: {args.seed}\n")
print(f"Data split: {(args.train_split, args.val_split, 1 - args.train_split - args.val_split)}")
print(f"Batch size: {(args.train_bs, args.val_bs)}\nSave names: {best_name}, {final_name}\n")

#==========#

#data: stage 1 runs once per split, stage 2 trains on crops from the cached patches
print("==========\n\nBegin dataset loading:\n")
split = MeasurementSplit(args.mdata, "box_images", (args.train_split, args.val_split), args.seed)
stage1 = load_stage1(args.s1, device)
segmenter = load_segmenter(device)
train_set = ShaftSet(split, "train", stage1, segmenter, device, scale = args.scale)
val_set = ShaftSet(split, "valid", stage1, segmenter, device, scale = args.scale)
del stage1, segmenter
torch.manual_seed(args.seed)

s1_pe = (torch.abs(val_set.s1_meas[:, :3] - val_set.y[:, :3]) / val_set.y[:, :3]).mean(dim = 0)
print(f"\nStage-1 percent error on M. cortical, L. cortical, shaft width (validation): {' '.join(f'{v:.4f}' for v in s1_pe.tolist())}\n")

#==========#

#model set up
model = ShaftKeypointNet().to(device)
if args.load:
    model.load_state_dict(torch.load(f"{save_dir}/{args.load}"))

#loss function: relative L1 on the three shaft measurements
lossfn = lambda ypred, yvals : torch.mean(torch.abs(ypred - yvals) / yvals)

#optimizer (halve the learning rate when validation stops improving; stage 2 converges within a few epochs)
optimizer = optim.Adam(model.parameters(), lr = args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = 2)

#training
early_stop_cnt = 0
best_val = float("inf")
os.makedirs(save_dir, exist_ok = True)

print("\n==========\n\nTraining started\n\n")
for epoch in range(args.epoch):
    model.train()
    order = torch.cat([torch.randperm(len(train_set)) for _ in range(args.repeats)])
    total_loss = 0

    for start in tqdm(range(0, len(order), args.train_bs), unit = "batch"):
        idx = order[start:start + args.train_bs]
        patches, centers, thetas, widths, yvals = train_set.batch(idx, device)
        crops, s = make_crops(patches, centers, thetas, widths, jitter = True)
        pos, _ = model(crops)

        loss = lossfn(positions_to_measurements(pos, s), yvals[:, :3])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(idx)

    print(f'Epoch: {epoch}\n')
    print(f'Loss (in sample): {total_loss / len(order)}')

    total_percent_err = torch.zeros(3, device = device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(val_set), args.val_bs):
            idx = torch.arange(start, min(start + args.val_bs, len(val_set)))
            patches, centers, thetas, widths, yvals = val_set.batch(idx, device)
            crops, s = make_crops(patches, centers, thetas, widths)
            pos, _ = model(crops)
            total_percent_err += torch.sum(torch.abs(positions_to_measurements(pos, s) - yvals[:, :3]) / yvals[:, :3], dim = 0)

    pe = total_percent_err / len(val_set)
    print(f'Percent error for M. cortical, L. cortical, shaft width (validation): {" ".join(f"{v:.4f}" for v in pe.tolist())}')
    print('\n===\n')

    val_metric = pe.mean().item()
    scheduler.step(val_metric)
    if val_metric > best_val:
        early_stop_cnt += 1
    else:
        early_stop_cnt = 0
        best_val = val_metric
        torch.save(model.state_dict(), f'{save_dir}/{best_name}')

    if early_stop_cnt >= args.esl:
        print('EARLY STOPPED')
        break

model.load_state_dict(torch.load(f"{save_dir}/{best_name}"))
torch.save(model.state_dict(), f"{save_dir}/{final_name}")
print(f"Best validation mean percent error (shaft measurements): {best_val:.4f}")
