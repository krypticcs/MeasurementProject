#imports
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from cnn_backbone.data_utils.data_utils import DataProcessor
from cnn_backbone.model.model import CNNModel
from common.geometry import PIX_PER_MM, model_to_coord, measurements_to_coord_points, coord_to_measurements
from common.losses import measurement_loss

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

parser = argparse.ArgumentParser(description = "Model training")

parser.add_argument("--mdata", type = str, required = True, help = "Path to measurement data file (from ./data/measurements)")
parser.add_argument("--idata", type = str, required = True, help = "Path to image data directory (from ./data)")

parser.add_argument("--aug", action = "store_true", help = "Random on-the-fly augmentation of the training set (validation is never augmented)")
parser.add_argument("--aug_repeats", type = int, default = 4, required = False, help = "Randomly augmented draws per image side per epoch (training set)")
parser.add_argument("--aug_rot", type = float, default = 10, required = False, help = "--aug: rotation sampled in [-aug_rot, aug_rot] degrees")
parser.add_argument("--aug_scale", type = float, default = 0.5, required = False, help = "--aug: image shrunk (zero-padded) by a factor sampled in [aug_scale, 1]")
parser.add_argument("--aug_crop", type = float, default = 0.6, required = False, help = "--aug: bottom-right fraction kept sampled in [aug_crop, 1]")
parser.add_argument("--freeze_enc", action = "store_true", help = "Freeze the pretrained autoencoder encoder (only train the head)")
parser.add_argument("--frozen_bn", action = "store_true", help = "With --freeze_enc: keep the encoder in eval mode while training (its BatchNorm statistics stay at the pretrained values)")

parser.add_argument("--epoch", type = int, default = 100, required = False, help = "Max number of epochs")
parser.add_argument("--lr", type = float, default = 1e-3, required = False, help = "Learning rate")
parser.add_argument("--esl", type = int, default = 5, required = False, help = "Number of epochs with no improvement to early stop training")
parser.add_argument("--meas_w", type = float, default = 0, required = False, help = "Weight of the measurement-space (relative error) loss added to the keypoint loss")
parser.add_argument("--select", type = str, default = "loss", required = False, help = "Checkpoint selection metric on validation (loss, pe = mean percent error)")
parser.add_argument("--run", type = str, default = "", required = False, help = "Run name: saves <run>_best.pth / <run>_final.pth instead of current_best.pth / final_model.pth")
parser.add_argument("--load", type = str, default = "", required = False, help = "Load a model to continue training (./cnn_backbone/model_saves)")
parser.add_argument("--seed", type = int, default = 42, required = False, help = "Torch seed")
parser.add_argument("--train_split", type = float, default = 0.8, required = False, help = "Training set split")
parser.add_argument("--val_split", type = float, default = 0.1, required = False, help = "Validation set split")
parser.add_argument("--train_bs", type = int, default = 64, required = False, help = "Training set batch size")
parser.add_argument("--val_bs", type = int, default = 64, required = False, help = "Validation set batch size")
parser.add_argument("--test_bs", type = int, default = 32, required = False, help = "Testiing set batch size")

args = parser.parse_args()

meas_w = args.meas_w
select_metric = args.select
best_name = f"{args.run}_best.pth" if args.run else "current_best.pth"
final_name = f"{args.run}_final.pth" if args.run else "final_model.pth"
if select_metric not in ["loss", "pe"]:
    raise ValueError(f"Unknown selection metric: {select_metric}")
print(f"Measurement loss weight: {meas_w}\nCheckpoint selection: {select_metric}\nSave names: {best_name}, {final_name}\n")

#settings

pix_per_mm = PIX_PER_MM
img_scale_factor = 0.1
img_width = int(2400 * img_scale_factor)
img_height = int(1920 * img_scale_factor)

measurement_file = args.mdata
img_dir = args.idata

aug_data = args.aug
freeze_enc = args.freeze_enc

epoch_cnt = args.epoch
learning_rate = args.lr
early_stop_lim = args.esl
load_path = None if args.load == "" else args.load
seed = args.seed
train_split = args.train_split
val_split = args.val_split
train_batch_size = args.train_bs
val_batch_size = args.val_bs
test_batch_size = args.test_bs
batch_sizes = (train_batch_size, val_batch_size, test_batch_size)

print("\nSelected settings:\n")
print(f"Measurement file: {measurement_file}\nImage directory: {img_dir}\n")
print(f"Augment data: {aug_data}" + (f" (rotation +-{args.aug_rot}, scale [{args.aug_scale}, 1], crop [{args.aug_crop}, 1], {args.aug_repeats} draws per femur per epoch)" if aug_data else "") + f"\nFreeze encoder: {freeze_enc}\nMax epoch cnt: {epoch_cnt}\nLearning rate: {learning_rate}\nEarly stopping limit: {early_stop_lim}\nLoad pretrained: {load_path}\nTorch seed: {seed}\n")
print(f"Data split: {(train_split, val_split, 1 - train_split - val_split)}")
print(f"Batch size: {batch_sizes}\n")

#==========#

#data
print("==========\n\nBegin dataset loading:\n")
data_processor = DataProcessor(measurement_file, img_dir, (train_split, val_split), batch_sizes, img_width, img_height, seed)
if aug_data:
    train_set, train_loader = data_processor.create_ds("train", (-args.aug_rot, args.aug_rot, 1), (args.aug_scale, 1, 1), (args.aug_crop, 1, 1), aug_repeats = args.aug_repeats) #(min, max, step) bounds; training samples uniformly in [min, max]
    val_set, val_loader = data_processor.create_ds("valid")
else:
    train_set, train_loader = data_processor.create_ds("train", aug_repeats = args.aug_repeats)
    val_set, val_loader = data_processor.create_ds("valid")

#==========#

#model set up
model = CNNModel(img_width, img_height, ae_ch_cnt = [1, 64, 128, 32, 1], device = device)
if load_path:
    model.load_state_dict(torch.load(f"./cnn_backbone/model_saves/{load_path}"))

if freeze_enc:
    for param in model.encoder.parameters():
        param.requires_grad = False

#loss function
lossfn = lambda ypred, yvals, aug_scales : torch.mean(torch.sum(torch.norm(ypred - yvals, dim = 2), dim = 1) / aug_scales)

#optimizer
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = learning_rate)

#training
early_stop_cnt = 0
best_val_loss = float("inf")
os.makedirs("./cnn_backbone/model_saves", exist_ok = True)

print("\n==========\n\nTraining started\n\n")
for epoch in range(epoch_cnt):
    model.to(device)
    model.train()
    if freeze_enc and args.frozen_bn:
        model.encoder.eval()

    total_loss = 0
    
    for images, yvals, aug_scales in tqdm(train_loader, unit = "batch"):
        images = images.to(device)
        yvals = yvals.to(device)
        aug_scales = aug_scales.to(device)

        model_out = model(images)
        model_coord, ab = model_to_coord(model_out)
        real_coord = measurements_to_coord_points(yvals, ab, pix_per_mm, img_scale_factor)

        loss = lossfn(model_coord, real_coord, aug_scales)
        if meas_w > 0:
            loss = loss + meas_w * measurement_loss(model_coord, yvals, pix_per_mm, img_scale_factor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.shape[0]

    print(f'Epoch: {epoch}\n')
    print(f'Loss (in sample): {total_loss / len(train_set)}')

    total_loss = 0
    total_percent_err = torch.zeros(10).to(device)

    model.eval()
    with torch.no_grad():
        for images, yvals, aug_scales in val_loader:
            images = images.to(device)
            yvals = yvals.to(device)
            aug_scales = aug_scales.to(device)

            model_out = model(images)
            model_coord, ab = model_to_coord(model_out)
            real_coord = measurements_to_coord_points(yvals, ab, pix_per_mm, img_scale_factor)

            loss = lossfn(model_coord, real_coord, aug_scales)
            if meas_w > 0:
                loss = loss + meas_w * measurement_loss(model_coord, yvals, pix_per_mm, img_scale_factor)

            total_loss += loss.item() * images.shape[0]

            total_percent_err += torch.sum(torch.abs(yvals - coord_to_measurements(model_coord, pix_per_mm, img_scale_factor)) / yvals, dim = 0)

        print(f'Loss (validation): {total_loss / len(val_set)}\n')
        print(f'Percent error for each measurement (validation):')
        for i in total_percent_err:
            print(f'{(i.item() / len(val_set)):.4f}', end = ' ')
        print()

        print('\n===\n')

    val_metric = total_loss if select_metric == "loss" else total_percent_err.sum().item() / (10 * len(val_set))
    if val_metric > best_val_loss:
        early_stop_cnt += 1
    else:
        early_stop_cnt = 0
        best_val_loss = val_metric
        torch.save(model.state_dict(), f'./cnn_backbone/model_saves/{best_name}')

    if early_stop_cnt >= early_stop_lim:
        print('EARLY STOPPED')
        break

model.load_state_dict(torch.load(f"./cnn_backbone/model_saves/{best_name}"))
torch.save(model.state_dict(), f"./cnn_backbone/model_saves/{final_name}")
