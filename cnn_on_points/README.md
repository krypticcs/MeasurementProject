# cnn_on_points

CNN that directly regresses 11 keypoint parameters from DXA images. Keypoints are converted to 10 femur measurements via geometric formulas.

## Architecture
- **SimpleCNNModel**: 4-layer CNN (32->64->128->256 channels) with MaxPool, flatten, linear (-> 4096 -> 11 outputs)
- **AlexNet**: AlexNet-style variant with larger kernels
- Optional batch norm (before/after ReLU, or none)
- Input: 192x240 grayscale (0.1x scale)

Keypoint/measurement geometry lives in `common/geometry.py` (shared with the other regression models).

## Usage
```bash
# train (run from project root)
python3 -m cnn_on_points.train --mdata 20260407.csv --idata box_images --model basic --esl 5

# test
python3 -m cnn_on_points.test --mdata 20260407.csv --idata box_images --ds test --model basic --makecsv

# lateral cortical diagnostic
python3 -m cnn_on_points.test_lateral_cortical --mdata 20260407.csv --idata box_images --ds test --model basic
```

Weights are saved to `cnn_on_points/model_saves/`, test outputs to `cnn_on_points/test_results/`.
