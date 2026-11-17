# seg_heatmaps

Heatmap model using a pretrained segmentation U-Net encoder as backbone. The seg encoder is frozen; only the decoder trains.

## Architecture
- **Encoder**: SegUNet encoder from `SEGMENTATIONS/mod_seg`, pretrained on femur segmentation, frozen
- **Decoder**: U-Net decoder with skip connections, outputs 13 keypoint heatmaps
- Soft-argmax to extract coordinates, then geometric conversion to 10 measurements
- Input: 192x240 grayscale (0.1x scale)

The decoder outputs raw logits (`SegUNet(13, final_act = False)`); the soft-argmax in `common/geometry.py` normalizes them, so there is no Sigmoid before it.

Requires `SEGMENTATIONS/mod_seg/model_saves/segunet.pth` (see "Required weights" in the root README).

## Usage
```bash
# train (run from project root)
python3 -m seg_heatmaps.train --mdata 20260407.csv --esl 5

# test (clean images only)
python3 -m seg_heatmaps.test --mdata 20260407.csv --ds test --makecsv

# test, also evaluating occluded copies of each image
python3 -m seg_heatmaps.test --mdata 20260407.csv --ds test --makecsv --mask
```

Only the training set gets random occlusion masking, with a fresh random amount on every draw (`--aug` adds random rotation/scale/crop on top; `--aug_repeats` draws per femur per epoch, default 4); validation is always evaluated on clean images. The loader is the shared one in `common/data.py`.
