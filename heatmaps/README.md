# heatmaps

Heatmap-based keypoint detection model for femur measurements. Predicts 13 keypoint heatmaps, converts to coordinates via soft-argmax, then to 10 measurements via geometry.

## Architecture
- 4-layer encoder (32->64->128->256) with stride-2 downsampling
- 1x1 conv to produce 13 heatmaps at reduced resolution
- Soft-argmax to extract (x, y) coordinates per keypoint
- Input: 192x240 grayscale (0.1x scale)

Soft-argmax and keypoint/measurement geometry live in `common/geometry.py` (shared with `seg_heatmaps` and `dual_heatmaps`).

## Usage
```bash
# train (run from project root)
python3 -m heatmaps.train --mdata 20260407.csv --idata box_images --model hm --esl 5

# test
python3 -m heatmaps.test --mdata 20260407.csv --idata box_images --ds test --model hm --makecsv

# occlusion test (masks progressively larger regions of each image)
python3 -m heatmaps.special_test --mdata 20260407.csv --idata box_images --ds test --model hm
```

During training, each training image gets random occlusion masking with a fresh random amount on every draw (a top/left band is zeroed, keeping only the bottom-right region; `--aug` adds random rotation/scale/crop on top; `--aug_repeats` draws per femur per epoch, default 4). Validation and test sets are never masked. The loader is the shared one in `common/data.py`.
