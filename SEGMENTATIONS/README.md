# SEGMENTATIONS

U-Net femur segmentation models. The `mod_seg` encoder is reused (frozen) by `seg_heatmaps`, `dual_heatmaps`, and `points_with_seg`.

- `original_seg/`: U-Net for the original public dataset (512x256)
- `mod_seg/`: U-Net for our images (192x240). `SegUNet(fin_ch, final_act = True)` -- pass `final_act = False` to drop the output Sigmoid (used by `seg_heatmaps`)
- `testing/`: runs the trained `mod_seg` model on unlabeled DXA images from `data/box_images` for visual inspection

Training data is read from `data/segmentation/data/xray/` and `data/segmentation/data/seg/`.

## Usage
```bash
# run from project root
python3 -m SEGMENTATIONS.mod_seg.train --esl 5
python3 -m SEGMENTATIONS.mod_seg.test --ds test

python3 -m SEGMENTATIONS.original_seg.train --esl 5
python3 -m SEGMENTATIONS.original_seg.test --ds test

python3 -m SEGMENTATIONS.testing.test --tccnt 20
```

Weights are saved to `SEGMENTATIONS/<model>/model_saves/`. The downstream models load `SEGMENTATIONS/mod_seg/model_saves/segunet.pth`, so copy the run you want to use there (see "Required weights" in the root README).
