# Results

All numbers are on the **test split** of `20260427.csv` with `--seed 42`: 1703 images have all 20 measurements
(1362 train / 170 validation / 171 test images), each image gives two femurs, so **n = 342** test femurs.
Every model is trained once (one seed), so differences of a few tenths of a percent are within run-to-run noise.

- Percent error = mean of |pred - true| / true. MAE is in mm (degrees for the neck-shaft angle).
- "null (train mean, same split)" predicts the training-set mean of each measurement and is scored on the same 342 femurs.
  `null_model/null_model.py` uses its own random 80/20 split; on `20260427.csv` it prints
  `0.1952 0.2005 0.0833 0.0775 0.1122 0.1077 0.0953 0.0807 0.0687 0.0528` (its NSA number differs because of that split).
- Reproduce a table with `python -m experiments.summarize --only <names>`; each name is a copy of a test script's
  `results.csv` in `experiments/results/`. Train/test logs are in `experiments/logs/` (gitignored).
- `experiments/run.ps1 -Folder <model> -Name <run> -TrainArgs "..." -TestArgs "..."` trains, tests and saves both.

Column key: M.Cort / L.Cort = medial / lateral cortical thickness, Shaft = shaft width, Head = femoral head diameter,
H.Off / V.Off = horizontal / vertical offset, Neck W = femoral neck width, HAL = hip axis length,
NAL = femoral neck axis length, NSA = neck-shaft angle.

**Summary (mean percent error over the 10 measurements).** Null model 10.62%. Best baselines: cnn_on_points 9.25%,
seg_heatmaps 9.57%. Measurement loss: seg_heatmaps `--meas_w 50` 8.35% (section 2). Mild on-the-fly augmentation:
cnn_on_points 8.66%, no gain for seg_heatmaps (section 3). Two-stage shaft crop on a seg_heatmaps stage 1: **7.03%**,
with medial/lateral cortical thickness and shaft width at 14.3% / 12.1% / 4.0% instead of 22.3% / 25.6% / 8.0% (section 4).

## 1. Baseline (original code, original loaders)

```
python -m cnn_on_points.train --mdata 20260427.csv --idata box_images --model basic --esl 5
python -m heatmaps.train      --mdata 20260427.csv --idata box_images --model hm --esl 5 --run baseline
python -m seg_heatmaps.train  --mdata 20260427.csv --esl 5 --run baseline
```

### Mean percent error

| Model | n | M.Cort | L.Cort | Shaft | Head | H.Off | V.Off | Neck W | HAL | NAL | NSA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null (train mean, same split) | 342 | 20.31% | 19.29% | 8.08% | 7.69% | 12.14% | 11.78% | 9.25% | 7.67% | 6.96% | 3.04% | 10.62% |
| cnn_on_points | 342 | 19.65% | 17.25% | 7.30% | 5.80% | 10.45% | 9.19% | 7.92% | 6.39% | 5.34% | 3.18% | 9.25% |
| heatmaps | 342 | 24.56% | 25.21% | 7.41% | 6.81% | 10.82% | 9.60% | 7.05% | 6.27% | 5.31% | 3.27% | 10.63% |
| seg_heatmaps | 342 | 22.32% | 25.57% | 8.00% | 5.18% | 8.58% | 7.98% | 6.03% | 4.95% | 4.02% | 3.12% | 9.57% |

### Mean absolute error

| Model | n | M.Cort | L.Cort | Shaft | Head | H.Off | V.Off | Neck W | HAL | NAL | NSA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null (train mean, same split) | 342 | 1.373 | 1.029 | 3.123 | 4.035 | 4.964 | 9.068 | 3.414 | 10.191 | 7.801 | 3.948 | 4.895 |
| cnn_on_points | 342 | 1.338 | 0.920 | 2.828 | 3.044 | 4.237 | 7.155 | 3.036 | 8.698 | 6.088 | 4.165 | 4.151 |
| heatmaps | 342 | 1.559 | 1.498 | 2.872 | 3.699 | 4.402 | 7.553 | 2.627 | 8.389 | 6.020 | 4.271 | 4.289 |
| seg_heatmaps | 342 | 1.621 | 1.273 | 2.981 | 2.718 | 3.442 | 5.932 | 2.211 | 6.445 | 4.443 | 4.007 | 3.507 |

### Std dev of absolute error

| Model | n | M.Cort | L.Cort | Shaft | Head | H.Off | V.Off | Neck W | HAL | NAL | NSA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null (train mean, same split) | 342 | 1.033 | 0.775 | 2.579 | 3.052 | 3.480 | 6.681 | 2.370 | 8.290 | 6.224 | 3.160 | 3.764 |
| cnn_on_points | 342 | 1.014 | 0.725 | 2.228 | 2.193 | 3.241 | 6.169 | 2.282 | 6.980 | 4.806 | 3.264 | 3.290 |
| heatmaps | 342 | 1.075 | 1.060 | 2.302 | 2.789 | 3.348 | 6.143 | 1.935 | 6.731 | 5.035 | 3.390 | 3.381 |
| seg_heatmaps | 342 | 1.200 | 0.917 | 2.391 | 2.041 | 2.904 | 5.168 | 1.716 | 4.849 | 3.417 | 2.815 | 2.742 |

**Takeaways.** `seg_heatmaps` is the best model for the head/neck/offset measurements (3-9), `cnn_on_points` has
the best overall mean only because it is better on the two cortical thicknesses. On the cortical thicknesses every
model is at or worse than the null model: at 0.1x a cortex is about 4 pixels wide, so the keypoints can't resolve it.

## 2. Measurement loss (`--meas_w`, `--select pe`)

`--meas_w w` adds `w * measurement_loss` (`common/losses.py`: mean relative L1 between the 10 measurements computed from
the predicted keypoints and the labels) to the keypoint loss. `--select pe` keeps the checkpoint with the lowest mean
validation percent error instead of the lowest validation loss. Same loaders, split and seed as section 1.

```
python -m cnn_on_points.train --mdata 20260427.csv --idata box_images --model basic --esl 5 --seed 42 --select pe [--meas_w 5|20|50]
python -m seg_heatmaps.train  --mdata 20260427.csv --esl 5 --seed 42 --select pe --meas_w 20|50
```

### Mean percent error

| Model | n | M.Cort | L.Cort | Shaft | Head | H.Off | V.Off | Neck W | HAL | NAL | NSA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null (train mean, same split) | 342 | 20.31% | 19.29% | 8.08% | 7.69% | 12.14% | 11.78% | 9.25% | 7.67% | 6.96% | 3.04% | 10.62% |
| cnn_on_points baseline | 342 | 19.65% | 17.25% | 7.30% | 5.80% | 10.45% | 9.19% | 7.92% | 6.39% | 5.34% | 3.18% | 9.25% |
| cnn_on_points select pe | 342 | 19.29% | 17.98% | 7.27% | 5.62% | 10.30% | 9.38% | 6.97% | 5.97% | 5.34% | 3.18% | 9.13% |
| cnn_on_points meas_w 5 | 342 | 19.62% | 17.73% | 7.40% | 6.09% | 10.86% | 9.49% | 7.97% | 6.16% | 5.43% | 3.14% | 9.39% |
| cnn_on_points meas_w 20 | 342 | 19.75% | 18.71% | 7.42% | 6.38% | 10.70% | 9.26% | 7.33% | 6.20% | 5.34% | 3.36% | 9.45% |
| cnn_on_points meas_w 50 | 342 | 20.52% | 17.43% | 7.51% | 6.00% | 10.73% | 9.59% | 7.31% | 6.12% | 5.34% | 3.10% | 9.37% |
| seg_heatmaps baseline | 342 | 22.32% | 25.57% | 8.00% | 5.18% | 8.58% | 7.98% | 6.03% | 4.95% | 4.02% | 3.12% | 9.57% |
| seg_heatmaps meas_w 20 | 342 | 22.45% | 19.08% | 7.96% | 5.86% | 8.74% | 8.46% | 7.51% | 5.04% | 4.21% | 3.29% | 9.26% |
| seg_heatmaps meas_w 50 | 342 | 20.35% | 17.76% | 6.67% | 4.93% | 8.28% | 7.65% | 5.87% | 4.94% | 3.99% | 3.02% | **8.35%** |

### Mean absolute error

| Model | n | M.Cort | L.Cort | Shaft | Head | H.Off | V.Off | Neck W | HAL | NAL | NSA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null (train mean, same split) | 342 | 1.373 | 1.029 | 3.123 | 4.035 | 4.964 | 9.068 | 3.414 | 10.191 | 7.801 | 3.948 | 4.895 |
| cnn_on_points baseline | 342 | 1.338 | 0.920 | 2.828 | 3.044 | 4.237 | 7.155 | 3.036 | 8.698 | 6.088 | 4.165 | 4.151 |
| cnn_on_points select pe | 342 | 1.377 | 0.933 | 2.859 | 2.961 | 4.222 | 7.300 | 2.567 | 8.050 | 6.096 | 4.164 | 4.053 |
| cnn_on_points meas_w 5 | 342 | 1.361 | 0.946 | 2.891 | 3.245 | 4.439 | 7.449 | 3.049 | 8.170 | 6.135 | 4.044 | 4.173 |
| cnn_on_points meas_w 20 | 342 | 1.443 | 1.115 | 2.901 | 3.472 | 4.434 | 7.311 | 2.680 | 8.087 | 5.945 | 4.297 | 4.169 |
| cnn_on_points meas_w 50 | 342 | 1.362 | 1.007 | 2.892 | 3.171 | 4.392 | 7.341 | 2.777 | 8.200 | 6.000 | 4.012 | 4.115 |
| seg_heatmaps baseline | 342 | 1.621 | 1.273 | 2.981 | 2.718 | 3.442 | 5.932 | 2.211 | 6.445 | 4.443 | 4.007 | 3.507 |
| seg_heatmaps meas_w 20 | 342 | 1.719 | 1.049 | 3.136 | 3.031 | 3.749 | 6.836 | 2.841 | 6.723 | 4.740 | 4.214 | 3.804 |
| seg_heatmaps meas_w 50 | 342 | 1.518 | 1.014 | 2.592 | 2.591 | 3.433 | 6.043 | 2.173 | 6.678 | 4.547 | 3.901 | 3.449 |

### Std dev of absolute error

| Model | n | M.Cort | L.Cort | Shaft | Head | H.Off | V.Off | Neck W | HAL | NAL | NSA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null (train mean, same split) | 342 | 1.033 | 0.775 | 2.579 | 3.052 | 3.480 | 6.681 | 2.370 | 8.290 | 6.224 | 3.160 | 3.764 |
| cnn_on_points baseline | 342 | 1.014 | 0.725 | 2.228 | 2.193 | 3.241 | 6.169 | 2.282 | 6.980 | 4.806 | 3.264 | 3.290 |
| cnn_on_points select pe | 342 | 1.064 | 0.720 | 2.302 | 2.247 | 3.171 | 6.094 | 1.999 | 6.476 | 4.847 | 3.261 | 3.218 |
| cnn_on_points meas_w 5 | 342 | 1.034 | 0.726 | 2.269 | 2.445 | 3.328 | 6.397 | 2.286 | 6.676 | 4.905 | 3.136 | 3.320 |
| cnn_on_points meas_w 20 | 342 | 1.104 | 0.845 | 2.289 | 2.708 | 3.270 | 6.315 | 1.970 | 6.522 | 4.759 | 3.263 | 3.304 |
| cnn_on_points meas_w 50 | 342 | 1.006 | 0.803 | 2.268 | 2.250 | 3.374 | 6.268 | 2.179 | 6.575 | 4.767 | 3.110 | 3.260 |
| seg_heatmaps baseline | 342 | 1.200 | 0.917 | 2.391 | 2.041 | 2.904 | 5.168 | 1.716 | 4.849 | 3.417 | 2.815 | 2.742 |
| seg_heatmaps meas_w 20 | 342 | 1.322 | 0.813 | 2.473 | 2.201 | 2.880 | 5.476 | 1.979 | 5.049 | 3.680 | 2.942 | 2.881 |
| seg_heatmaps meas_w 50 | 342 | 1.140 | 0.806 | 2.120 | 2.096 | 2.739 | 5.190 | 1.573 | 5.195 | 3.689 | 2.713 | 2.726 |

**Takeaways.** For `cnn_on_points` the measurement loss does not help at any weight (9.37-9.45% vs 9.13-9.25%): it
regresses keypoint coordinates directly, so the keypoint loss and the measurement loss already point the same way.
For `seg_heatmaps` `--meas_w 50` is the best single model so far (8.35%) and improves 9 of 10 measurements over its
baseline, mostly the cortical thicknesses (lateral 25.6% -> 17.8%) and the shaft width. The heatmap loss only asks each
keypoint to be near its label; the measurement loss directly penalises the small W/X and Y/Z gaps that the
cortical thicknesses depend on. `meas_w 20` was in between (lateral improved, the neck measurements got worse), so
treat the ordering between nearby weights as noisy. `--meas_w 50` is used for seg_heatmaps in the next section.

## 3. On-the-fly augmentation (`--aug`, shared loader in `common/data.py`)

Every keypoint model now loads data through `common/data.py`. Each image is cached once at 2x the target resolution,
and every training draw gets fresh random augmentation: rotation in [-r, r], scale in [s, 1], crop in [c, 1]
(plus the occlusion mask in [0, 0.5] for the heatmap models, as before). Validation and test images are never augmented.
`--aug_repeats 4` draws per femur per epoch. With augmentation off, the new loader reproduces the old `DataPoint` split,
labels and images (`python -m experiments.check_loader_equivalence`: identical split and labels, mean pixel difference
0.007 on a [0, 1] scale from the different resize path).

- **full**: the plan's ranges, `--aug` (rotation +-10, scale [0.5, 1], crop [0.6, 1]).
- **mild**: `--aug --aug_rot 5 --aug_scale 0.8 --aug_crop 0.85`.
- **frozen BN**: `--frozen_bn` keeps the frozen pretrained encoder in eval mode during training (see the takeaways).

All runs below use `--select pe`. The seg_heatmaps augmentation runs use `--esl 10` when `--frozen_bn` is set
(validation error is noisier with augmentation) and `--esl 5` otherwise.

```
python -m cnn_on_points.train --mdata 20260427.csv --idata box_images --model basic --esl 5 --seed 42 --select pe --aug [--aug_rot 5 --aug_scale 0.8 --aug_crop 0.85] [--meas_w 50]
python -m seg_heatmaps.train  --mdata 20260427.csv --seed 42 --select pe --aug [--aug_rot 5 --aug_scale 0.8 --aug_crop 0.85] [--meas_w 50] [--frozen_bn --esl 10]
```

### Mean percent error

| Model | n | M.Cort | L.Cort | Shaft | Head | H.Off | V.Off | Neck W | HAL | NAL | NSA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null (train mean, same split) | 342 | 20.31% | 19.29% | 8.08% | 7.69% | 12.14% | 11.78% | 9.25% | 7.67% | 6.96% | 3.04% | 10.62% |
| cnn_on_points select pe (no aug) | 342 | 19.29% | 17.98% | 7.27% | 5.62% | 10.30% | 9.38% | 6.97% | 5.97% | 5.34% | 3.18% | 9.13% |
| cnn_on_points aug full | 342 | 19.86% | 18.02% | 7.24% | 5.82% | 9.42% | 9.30% | 7.39% | 6.07% | 5.49% | 3.25% | 9.19% |
| cnn_on_points aug full + meas_w 50 | 342 | 20.93% | 20.28% | 8.73% | 7.99% | 12.16% | 10.58% | 9.03% | 8.32% | 7.63% | 3.16% | 10.88% |
| cnn_on_points aug mild | 342 | 19.82% | 17.27% | 6.44% | 5.29% | 9.05% | 9.25% | 6.32% | 5.45% | 4.64% | 3.06% | **8.66%** |
| seg_heatmaps meas_w 50 (no aug) | 342 | 20.35% | 17.76% | 6.67% | 4.93% | 8.28% | 7.65% | 5.87% | 4.94% | 3.99% | 3.02% | **8.35%** |
| seg_heatmaps aug full | 342 | 24.15% | 28.86% | 8.58% | 10.78% | 22.83% | 8.78% | 10.85% | 7.23% | 6.46% | 3.91% | 13.24% |
| seg_heatmaps aug full + meas_w 50 | 342 | 35.73% | 21.35% | 20.90% | 9.67% | 19.41% | 10.41% | 13.21% | 8.71% | 8.36% | 4.25% | 15.20% |
| seg_heatmaps aug full + meas_w 50 + frozen BN | 342 | 20.50% | 19.73% | 6.74% | 8.77% | 15.91% | 7.21% | 8.69% | 5.91% | 6.27% | 4.72% | 10.44% |
| seg_heatmaps aug mild + meas_w 50 + frozen BN | 342 | 19.77% | 21.86% | 6.66% | 5.19% | 12.14% | 7.38% | 7.46% | 4.66% | 4.80% | 3.30% | 9.32% |

### Mean absolute error

| Model | n | M.Cort | L.Cort | Shaft | Head | H.Off | V.Off | Neck W | HAL | NAL | NSA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null (train mean, same split) | 342 | 1.373 | 1.029 | 3.123 | 4.035 | 4.964 | 9.068 | 3.414 | 10.191 | 7.801 | 3.948 | 4.895 |
| cnn_on_points select pe (no aug) | 342 | 1.377 | 0.933 | 2.859 | 2.961 | 4.222 | 7.300 | 2.567 | 8.050 | 6.096 | 4.164 | 4.053 |
| cnn_on_points aug full | 342 | 1.480 | 1.002 | 2.830 | 3.090 | 3.934 | 7.203 | 2.735 | 8.051 | 6.176 | 4.166 | 4.067 |
| cnn_on_points aug full + meas_w 50 | 342 | 1.433 | 1.220 | 3.393 | 4.217 | 5.014 | 8.197 | 3.351 | 11.089 | 8.569 | 4.069 | 5.055 |
| cnn_on_points aug mild | 342 | 1.321 | 0.958 | 2.521 | 2.838 | 3.723 | 7.345 | 2.382 | 7.374 | 5.292 | 3.951 | 3.771 |
| seg_heatmaps meas_w 50 (no aug) | 342 | 1.518 | 1.014 | 2.592 | 2.591 | 3.433 | 6.043 | 2.173 | 6.678 | 4.547 | 3.901 | 3.449 |
| seg_heatmaps aug full | 342 | 1.779 | 1.427 | 3.373 | 5.716 | 9.502 | 6.849 | 4.035 | 9.741 | 7.344 | 5.070 | 5.484 |
| seg_heatmaps aug full + meas_w 50 | 342 | 2.801 | 1.247 | 8.320 | 5.116 | 8.116 | 8.413 | 4.883 | 11.721 | 9.551 | 5.490 | 6.566 |
| seg_heatmaps aug full + meas_w 50 + frozen BN | 342 | 1.521 | 1.004 | 2.648 | 4.688 | 6.567 | 5.711 | 3.227 | 7.921 | 7.135 | 6.195 | 4.662 |
| seg_heatmaps aug mild + meas_w 50 + frozen BN | 342 | 1.431 | 1.329 | 2.611 | 2.761 | 4.915 | 5.748 | 2.731 | 6.209 | 5.429 | 4.343 | 3.751 |

### Std dev of absolute error

| Model | n | M.Cort | L.Cort | Shaft | Head | H.Off | V.Off | Neck W | HAL | NAL | NSA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null (train mean, same split) | 342 | 1.033 | 0.775 | 2.579 | 3.052 | 3.480 | 6.681 | 2.370 | 8.290 | 6.224 | 3.160 | 3.764 |
| cnn_on_points select pe (no aug) | 342 | 1.064 | 0.720 | 2.302 | 2.247 | 3.171 | 6.094 | 1.999 | 6.476 | 4.847 | 3.261 | 3.218 |
| cnn_on_points aug full | 342 | 1.124 | 0.765 | 2.151 | 2.652 | 2.991 | 6.302 | 2.139 | 6.570 | 4.803 | 3.119 | 3.262 |
| cnn_on_points aug full + meas_w 50 | 342 | 1.047 | 0.925 | 2.791 | 3.037 | 3.770 | 6.718 | 2.381 | 8.129 | 6.830 | 3.250 | 3.888 |
| cnn_on_points aug mild | 342 | 0.995 | 0.748 | 2.154 | 2.434 | 2.938 | 6.280 | 2.048 | 5.909 | 4.652 | 2.876 | 3.104 |
| seg_heatmaps meas_w 50 (no aug) | 342 | 1.140 | 0.806 | 2.120 | 2.096 | 2.739 | 5.190 | 1.573 | 5.195 | 3.689 | 2.713 | 2.726 |
| seg_heatmaps aug full | 342 | 1.411 | 1.116 | 2.681 | 4.857 | 6.329 | 5.457 | 3.295 | 9.385 | 6.813 | 3.800 | 4.514 |
| seg_heatmaps aug full + meas_w 50 | 342 | 1.758 | 0.899 | 4.094 | 4.407 | 8.208 | 6.756 | 3.557 | 10.439 | 8.144 | 4.908 | 5.317 |
| seg_heatmaps aug full + meas_w 50 + frozen BN | 342 | 1.154 | 0.770 | 2.091 | 3.348 | 5.986 | 5.256 | 2.202 | 7.256 | 6.871 | 4.295 | 3.923 |
| seg_heatmaps aug mild + meas_w 50 + frozen BN | 342 | 1.049 | 0.942 | 2.192 | 2.111 | 3.966 | 4.883 | 2.278 | 4.705 | 3.857 | 3.284 | 2.927 |

**Takeaways.** The test images all have the same scale and orientation as the training images, so augmentation
only helps when it is mild enough not to move training away from the test distribution.
- `cnn_on_points` with mild augmentation is its best result (8.66% vs 9.13%) and improves 8 of 10 measurements; the
  full ranges are neutral (9.19%). Adding the measurement loss on top of augmentation hurts it, as it did without augmentation.
- `seg_heatmaps` gets worse with augmentation at every setting tried; the best is 9.32% (mild, frozen BN) vs 8.35% without.
  Most of the loss is in horizontal offset (8.3% -> 12.1%), neck width and the lateral cortical thickness.
- The full-range seg_heatmaps runs had validation error swinging between about 14% and 32% from one epoch to the next.
  Part of the cause: `requires_grad = False` freezes the pretrained `segunet` encoder's weights, but in `model.train()` its
  BatchNorm layers keep updating their running statistics, which drift toward the shrunken, zero-padded augmented
  images. `--frozen_bn` (now available in `seg_heatmaps`, `dual_heatmaps`, `pretrained_cnn` and `cnn_backbone --freeze_enc`)
  keeps them at the pretrained values and cut the error from 15.2% to 10.4% (that run also used `--esl 10` instead of 5;
  with `--frozen_bn` the best epoch was 32 of 43, while the run without it stopped after 9 epochs with validation error still swinging). Re-estimating the statistics after training
  does not help (the decoder has already adapted to them), so the flag has to be set while training. It defaults to off
  so the earlier sections stay reproducible.

## 4. Two-stage shaft crop model (`two_stage/`)

Stage 1 is a frozen `seg_heatmaps` checkpoint that gives all 10 measurements from the 0.1x image. Stage 2 refines
medial/lateral cortical thickness and shaft width from a rotated 64 x 256 crop of the femoral shaft taken from the 0.5x
image cache, where a cortex is about 20 pixels wide instead of about 4. Measurements 3-9 are stage 1's, unchanged.
See `two_stage/README.md` for the pipeline.

Deviation from the plan: the crop is not centred on stage 1's absolute W and Z keypoints. The heatmap models are trained
only on keypoint positions relative to the head centre A, so their absolute keypoints float, usually somewhere in the
middle of the pelvis (`experiments/stage1_absolute_keypoints.png`). The shaft is located from the femur segmentation
mask instead (`segunet.pth`), at stage 1's predicted vertical offset below the head centre, and stage 1's predicted
shaft width sets the crop scale.

```
python -m common.cache_images --mdata 20260427.csv
python -m two_stage.train --mdata 20260427.csv --s1 baseline_best.pth --run s1baselinelr
python -m two_stage.test  --mdata 20260427.csv --ds test --makecsv --s1 baseline_best.pth --path s1baselinelr_best.pth
python -m two_stage.train --mdata 20260427.csv --s1 measw50_best.pth --run s1measw50
python -m two_stage.test  --mdata 20260427.csv --ds test --makecsv --s1 measw50_best.pth --path s1measw50_best.pth
```

### Mean percent error

| Model | n | M.Cort | L.Cort | Shaft | Head | H.Off | V.Off | Neck W | HAL | NAL | NSA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null (train mean, same split) | 342 | 20.31% | 19.29% | 8.08% | 7.69% | 12.14% | 11.78% | 9.25% | 7.67% | 6.96% | 3.04% | 10.62% |
| seg_heatmaps baseline (stage 1) | 342 | 22.32% | 25.57% | 8.00% | 5.18% | 8.58% | 7.98% | 6.03% | 4.95% | 4.02% | 3.12% | 9.57% |
| two_stage on seg_heatmaps baseline | 342 | 14.31% | 12.08% | 4.02% | 5.18% | 8.58% | 7.98% | 6.03% | 4.95% | 4.02% | 3.12% | **7.03%** |
| seg_heatmaps meas_w 50 (stage 1) | 342 | 20.35% | 17.76% | 6.67% | 4.93% | 8.28% | 7.65% | 5.87% | 4.94% | 3.99% | 3.02% | 8.35% |
| two_stage on seg_heatmaps meas_w 50 | 342 | 15.57% | 14.16% | 4.01% | 4.93% | 8.28% | 7.65% | 5.87% | 4.94% | 3.99% | 3.02% | 7.24% |

### Mean absolute error

| Model | n | M.Cort | L.Cort | Shaft | Head | H.Off | V.Off | Neck W | HAL | NAL | NSA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null (train mean, same split) | 342 | 1.373 | 1.029 | 3.123 | 4.035 | 4.964 | 9.068 | 3.414 | 10.191 | 7.801 | 3.948 | 4.895 |
| seg_heatmaps baseline (stage 1) | 342 | 1.621 | 1.273 | 2.981 | 2.718 | 3.442 | 5.932 | 2.211 | 6.445 | 4.443 | 4.007 | 3.507 |
| two_stage on seg_heatmaps baseline | 342 | 0.976 | 0.680 | 1.558 | 2.718 | 3.442 | 5.933 | 2.211 | 6.445 | 4.443 | 4.007 | 3.241 |
| seg_heatmaps meas_w 50 (stage 1) | 342 | 1.518 | 1.014 | 2.592 | 2.591 | 3.433 | 6.043 | 2.173 | 6.678 | 4.547 | 3.901 | 3.449 |
| two_stage on seg_heatmaps meas_w 50 | 342 | 1.075 | 0.797 | 1.566 | 2.591 | 3.433 | 6.042 | 2.173 | 6.678 | 4.547 | 3.901 | 3.280 |

### Std dev of absolute error

| Model | n | M.Cort | L.Cort | Shaft | Head | H.Off | V.Off | Neck W | HAL | NAL | NSA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null (train mean, same split) | 342 | 1.033 | 0.775 | 2.579 | 3.052 | 3.480 | 6.681 | 2.370 | 8.290 | 6.224 | 3.160 | 3.764 |
| seg_heatmaps baseline (stage 1) | 342 | 1.200 | 0.917 | 2.391 | 2.041 | 2.904 | 5.168 | 1.716 | 4.849 | 3.417 | 2.815 | 2.742 |
| two_stage on seg_heatmaps baseline | 342 | 0.820 | 0.611 | 1.392 | 2.041 | 2.904 | 5.168 | 1.716 | 4.849 | 3.417 | 2.815 | 2.573 |
| seg_heatmaps meas_w 50 (stage 1) | 342 | 1.140 | 0.806 | 2.120 | 2.096 | 2.739 | 5.190 | 1.573 | 5.195 | 3.689 | 2.713 | 2.726 |
| two_stage on seg_heatmaps meas_w 50 | 342 | 0.877 | 0.654 | 1.357 | 2.096 | 2.739 | 5.189 | 1.573 | 5.195 | 3.689 | 2.713 | 2.608 |

**Takeaways.** The high-resolution crop is the biggest improvement in this project. It is the first model that beats the
null model on the cortical thicknesses by a clear margin (medial 20.3% -> 14.3%, lateral 19.3% -> 12.1%), and it
halves the shaft width error (8.0% -> 4.0%, MAE 2.98 -> 1.56 mm). Stage 2 does almost all of the work: on the
validation split the three stage-1/stage-2 pairings (baseline/baseline, meas_w 50/baseline, meas_w 50/meas_w 50) are
within 0.06 percentage points of each other (7.46-7.52% mean), so the gap between the two test rows above is mostly
run-to-run noise in stage 2. Measurements 3-9 are passed through from stage 1, so any better stage-1 model improves
them directly.

**Pixel spacing.** `common/geometry.py` assumes 0.17 mm/px for every image, but 64 of the 342 test femurs come from
images at 0.150 mm/px. If the labels were in calibrated mm these would come out about 13% too large. With the two-stage
model they come out 2.9% too large (median pred/true 1.029, vs 1.003 for the rest; 7.66% vs 6.88% mean percent error),
so the labels are mostly in the 0.17 mm/px convention and spacing explains only a small part of the error
(`python -m experiments.spacing_bias s1baselinelr_two_stage`).

### Two-stage by side: mean percent error

Same test results as above, split into the left (`L ...` columns of the measurement file) and right (`R ...` columns)
femur of each of the 171 test images.

| Model | Side | n | M.Cort | L.Cort | Shaft | Head | H.Off | V.Off | Neck W | HAL | NAL | NSA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| two_stage on seg_heatmaps baseline | Left | 171 | 15.28% | 11.76% | 3.82% | 5.30% | 8.49% | 8.60% | 5.70% | 4.62% | 3.83% | 3.21% | 7.06% |
| two_stage on seg_heatmaps baseline | Right | 171 | 13.34% | 12.40% | 4.21% | 5.05% | 8.67% | 7.35% | 6.36% | 5.29% | 4.22% | 3.04% | 6.99% |
| two_stage on seg_heatmaps meas_w 50 | Left | 171 | 15.73% | 13.71% | 3.89% | 4.90% | 8.29% | 8.58% | 5.44% | 4.89% | 4.02% | 3.05% | 7.25% |
| two_stage on seg_heatmaps meas_w 50 | Right | 171 | 15.41% | 14.62% | 4.13% | 4.97% | 8.26% | 6.72% | 6.29% | 4.99% | 3.95% | 2.99% | 7.23% |

### Two-stage by side: mean absolute error (mm, degrees for NSA)

| Model | Side | n | M.Cort | L.Cort | Shaft | Head | H.Off | V.Off | Neck W | HAL | NAL | NSA | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| two_stage on seg_heatmaps baseline | Left | 171 | 0.999 | 0.655 | 1.469 | 2.780 | 3.376 | 6.284 | 2.106 | 5.985 | 4.216 | 4.095 | 3.196 |
| two_stage on seg_heatmaps baseline | Right | 171 | 0.952 | 0.705 | 1.648 | 2.656 | 3.508 | 5.581 | 2.317 | 6.904 | 4.670 | 3.919 | 3.286 |
| two_stage on seg_heatmaps meas_w 50 | Left | 171 | 1.044 | 0.755 | 1.510 | 2.553 | 3.387 | 6.690 | 2.029 | 6.591 | 4.583 | 3.930 | 3.307 |
| two_stage on seg_heatmaps meas_w 50 | Right | 171 | 1.107 | 0.838 | 1.621 | 2.629 | 3.478 | 5.395 | 2.318 | 6.765 | 4.511 | 3.873 | 3.254 |
