# two_stage

Refines the three shaft measurements (medial cortical thickness, lateral cortical thickness, shaft width) from a
high-resolution crop of the femoral shaft. The other seven measurements come unchanged from stage 1.

At 0.1x scale (what every other model sees) a cortex is about 4 pixels wide. The crop is taken from the 0.5x image,
where it is about 20 pixels wide.

## Pipeline

1. **Stage 1 (frozen):** a trained `seg_heatmaps` model on the 0.1x image gives all 10 measurements.
2. **Shaft localization (frozen):** the femur segmenter `SEGMENTATIONS/mod_seg/model_saves/segunet.pth` outlines the
   femur being measured (for both sides it is the femur on the right of the possibly flipped image).
   - head center = deepest point of the mask medial to the shaft and within one (stage-1) head diameter of the top of the
     femur (further down, the intertrochanteric region is thicker than the head and would win),
   - shaft axis = line through the mask's row midpoints along the lower shaft,
   - the W-Z line is centered on the axis, stage 1's predicted vertical offset below the head center, perpendicular to the axis
     (the same definition `common/geometry.py` uses: A is the head center and V. offset is the drop from A to the W-Z line),
     and pulled back up the axis if it would fall below the visible femur (about 6% of femurs, where the shaft leaves the image).

   The heatmap models' absolute keypoints can't be used for this: they are only trained on keypoints *relative to A*,
   so the whole constellation floats (in practice it sits in the middle of the pelvis, see
   `experiments/stage1_absolute_keypoints.png`).
3. **Crop:** a rotated 64 x 256 crop from the 0.5x cache (`common/cache_images.py`, `data/cache/half/`) with columns along
   the W -> Z direction spanning 2x stage 1's predicted shaft width (rows use the same pixel scale). During training the
   center (+-10% of the width across, +-16 px along the shaft), angle (+-5 degrees) and scale (0.85-1.15) are jittered to
   simulate stage-1 error.
4. **Stage 2** (`model/model.py`, `ShaftKeypointNet`): ImageNet ResNet-18 stem (conv1 to layer2), features mean-pooled over
   height into a 1D profile, a small 1D U-Net decoder, 4 1D heatmaps and soft-argmax for the x-positions of W, X, Y, Z.
5. **Loss:** relative L1 on `[X - W, Z - Y, Z - W]` converted to mm against true measurements 0, 1 and 2. Only measurement
   differences are supervised (there are no absolute keypoint labels).

## Usage
```bash
# cache the 0.5x images once (about 3.9 GB for 20260427.csv; also built on demand)
python -m common.cache_images --mdata 20260427.csv

# train stage 2 on top of a trained seg_heatmaps model (from ./seg_heatmaps/model_saves/)
python -m two_stage.train --mdata 20260427.csv --s1 baseline_best.pth --run s1baseline

# test: writes two_stage/test_results/results.csv (same format as the other models) and prints stage-1-only errors too
python -m two_stage.test --mdata 20260427.csv --ds test --makecsv --s1 baseline_best.pth --path s1baseline_best.pth
```

Flags: `--repeats` (jittered crops per femur per epoch, default 4), `--esl` (default 10), `--lr` (default 5e-4, halved after 2 epochs without
validation improvement; stage 2 converges in a few epochs and the validation error is noisy without it).
Checkpoints are selected on mean validation percent error of the three shaft measurements.
`test.py` also plots the first `--n_img` crops with the predicted W/X/Y/Z positions to `two_stage/test_results/img/`.

Requires `segunet.pth` and a trained `seg_heatmaps` checkpoint (see "Required weights" in the root README).
