# points_with_seg

Keypoint regression model that works on a bone segmentation instead of the raw image. A frozen, pretrained `SEGMENTATIONS/mod_seg` U-Net turns each DXA image into a binary femur mask (threshold 0.5), and a CNN regresses 11 keypoint parameters from that mask alone -- the raw image intensities are not passed to the CNN.

## Architecture
- **Seg model**: `SegUNet(1)` loaded from `SEGMENTATIONS/mod_seg/model_saves/segunet.pth`, frozen
- **CNN**: 4-layer CNN (32->64->128->256 channels) with MaxPool, flatten, linear (-> 4096 -> 11 outputs), optional batch norm
- Same coordinate-to-measurement geometry as cnn_on_points (`common/geometry.py`)
- Input: 192x240 grayscale (0.1x scale)

Requires `SEGMENTATIONS/mod_seg/model_saves/segunet.pth` (see "Required weights" in the root README).

## Usage
```bash
# train (run from project root)
python3 -m points_with_seg.train --mdata 20260407.csv --idata box_images --esl 5

# train, excluding keypoints from the loss (indices into D, F, G, H, W, X, Y, Z, H', A', A'', C)
python3 -m points_with_seg.train --mdata 20260407.csv --idata box_images --esl 5 --exclude_pts 1

# test (saves raw image / seg mask / masked image / keypoints plots)
python3 -m points_with_seg.test --mdata 20260407.csv --idata box_images --ds test --makecsv

# lateral cortical diagnostic
python3 -m points_with_seg.test_lateral_cortical --mdata 20260407.csv --idata box_images --ds test
```

Weights are saved to `points_with_seg/model_saves/`, test outputs to `points_with_seg/test_results/`.
