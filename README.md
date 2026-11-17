# MeasurementProject

Automated femur measurement prediction from hip DXA images. Multiple deep learning approaches to predict 10 anatomical measurements per femur side (medial/lateral cortical thickness, shaft width, femoral head diameter, horizontal/vertical offset, neck width, hip axis length, neck axis length, neck-shaft angle).

## Active Models

| Directory | Approach | Description |
|---|---|---|
| `cnn_on_points/` | Direct regression | CNN regresses 11 keypoint params, converted to measurements via geometry |
| `heatmaps/` | Heatmap keypoints | Encoder produces 13 heatmaps, soft-argmax to coordinates |
| `seg_heatmaps/` | Seg backbone + heatmaps | Frozen pretrained segmentation encoder + heatmap decoder |
| `dual_heatmaps/` | Dual encoder + heatmaps | Frozen seg encoder + trainable raw encoder with fused skip connections |
| `points_with_seg/` | Seg mask + regression | Frozen seg model produces a binary femur mask; CNN regresses keypoints from the mask |
| `two_stage/` | Heatmaps + shaft crop | Frozen `seg_heatmaps` + femur mask locate the shaft; a 1D keypoint net on a 0.5x crop refines cortical thickness and shaft width |

## Other Models

| Directory | Approach |
|---|---|
| `autoencoder/` | U-Net autoencoder for image reconstruction (used as backbone) |
| `cnn_backbone/` | Pretrained autoencoder encoder + CNN head (`--freeze_enc` to freeze the encoder) |
| `keypoint_detection/` | Self-supervised keypoint detection via image reconstruction |
| `pretrained_cnn/` | Frozen DenseNet-121 (torchxrayvision) backbone + CNN head |
| `null_model/` | Baseline: predicts training set mean |
| `SEGMENTATIONS/` | U-Net segmentation models (original 512x256, modified 192x240) |
| `NONACTIVE_MODELS/` | Archived/deprecated approaches |

## Shared Code

`common/geometry.py` holds the keypoint/measurement geometry used by every keypoint model:
- Regression models (`cnn_on_points`, `cnn_backbone`, `pretrained_cnn`, `points_with_seg`): `model_to_coord`, `measurements_to_coord_points`, `coord_to_measurements`
- Heatmap models (`heatmaps`, `seg_heatmaps`, `dual_heatmaps`): `heatmaps_to_keypoints` (soft-argmax), `abs_kp_to_coord`, `get_ab`, `measurements_to_coord_heatmap`, `coord_to_measurements(..., flip_y = True)`
- Constants: `PIX_PER_MM`, `MEASUREMENT_NAMES`, keypoint label lists

`common/data.py` is the data loader behind every keypoint model's `data_utils.DataProcessor` (each is a thin wrapper):
- `MeasurementSplit`: CSV cleaning and the seeded `torch.randperm` train/valid/test split
- `CachedImages`: each DICOM is read once, normalized and kept as float16 at 2x the model resolution, in RAM and in `data/cache/`
- `AugDataset`: the training set gets fresh random rotation/scale/crop (and occlusion mask for the heatmap models) on every draw; validation and test are never augmented

`common/losses.py` has `measurement_loss` (relative error of the 10 measurements implied by the keypoints).
`common/cache_images.py` writes the 0.5x image cache used by `two_stage` (`python -m common.cache_images --mdata 20260427.csv`).

## Running

Every script is run from the project root as a module, and all paths (`./data/...`, `./<model>/model_saves/...`, `./<model>/test_results/...`) are relative to the root:

```bash
python3 -m <folder>.train --mdata 20260407.csv --idata box_images ...
python3 -m <folder>.test  --mdata 20260407.csv --idata box_images --ds test ...
python3 -m SEGMENTATIONS.mod_seg.train ...
python3 -m null_model.null_model
```

See each folder's README for the exact flags. Output folders (`model_saves/`, `test_results/`) are created automatically.

Useful flags:
- `--mask` (`seg_heatmaps.test`, `dual_heatmaps.test`): also evaluate occluded copies of each image. Without it, testing uses clean images only. Training-time occlusion masking is applied to the training set only.
- `--freeze_enc` (`cnn_backbone.train`): freeze the pretrained autoencoder encoder and train only the head.
- `--frozen_bn` (`seg_heatmaps`, `dual_heatmaps`, `pretrained_cnn`, `cnn_backbone --freeze_enc`): keep the frozen encoder in eval mode during training, so its BatchNorm statistics stay at the pretrained values instead of drifting toward the (augmented) training images.

Training flags shared by the 7 keypoint models (`cnn_on_points`, `cnn_backbone`, `pretrained_cnn`, `points_with_seg`, `heatmaps`, `seg_heatmaps`, `dual_heatmaps`):
- `--aug`: random on-the-fly augmentation of the training set (rotation in [-10, 10], scale in [0.5, 1], crop in [0.6, 1]; change the bounds with `--aug_rot`, `--aug_scale`, `--aug_crop`); `--aug_repeats` augmented draws per femur per epoch (default 4)
- `--meas_w`: weight of `measurement_loss` added to the keypoint loss (default 0)
- `--select {loss,pe}`: keep the checkpoint with the lowest validation loss (default) or mean validation percent error
- `--run <name>`: save `<name>_best.pth` / `<name>_final.pth` instead of `current_best.pth` / `final_model.pth` (test with `--path <name>_best.pth`)

## Experiments

`experiments/RESULTS.md` has the baseline and every comparison on the seed-42 test split of `20260427.csv`.
`experiments/run.ps1` trains + tests one configuration and keeps its logs (`experiments/logs/`) and `results.csv`
(`experiments/results/`); `python -m experiments.summarize` turns those into tables, next to a training-mean null model on the same split.

## Required Weights

Model weights are gitignored. Some models depend on weights from other models, at fixed paths:

| Needed by | File | How to produce it |
|---|---|---|
| `seg_heatmaps`, `dual_heatmaps`, `points_with_seg`, `two_stage` | `SEGMENTATIONS/mod_seg/model_saves/segunet.pth` | `python3 -m SEGMENTATIONS.mod_seg.train`, then copy `SEGMENTATIONS/mod_seg/model_saves/final_model.pth` to `segunet.pth` |
| `cnn_backbone` | `autoencoder/model_saves/autoencoder.pth` | `python3 -m autoencoder.train --idata box_images` (non-residual, channels `[1, 64, 128, 32, 1]`), then copy `autoencoder/model_saves/final_model.pth` to `autoencoder.pth` |
| `pretrained_cnn` | torchxrayvision DenseNet weights | Downloaded automatically by `torchxrayvision` on first use |
| `two_stage` | a trained `seg_heatmaps/model_saves/<name>.pth` (stage 1, `--s1 <name>.pth`) and ImageNet ResNet-18 | `python3 -m seg_heatmaps.train ... --run <name>`; ResNet-18 weights download automatically |

Test scripts load `<model>/model_saves/current_best.pth` by default (override with `--path`).

## Data
- `data/measurements/` -- Dated CSVs of human-annotated measurements (~3094 images)
- `data/box_images/` -- DXA DICOM images (gitignored)
- `data/segmentation/` -- Segmentation masks (gitignored)
