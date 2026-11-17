# pretrained_cnn

Measurement model using a pretrained DenseNet-121 backbone from torchxrayvision (trained on chest X-rays). The backbone is frozen; additional conv + linear layers are trained to regress 11 keypoint parameters.

## Architecture
- **Backbone**: First 7 feature blocks of `xrv.models.DenseNet(weights="densenet121-res224-all")` -> 512 channels at 28x28 (frozen)
- **Head**: Conv 512->256->256, MaxPool, Flatten, Linear -> 256 -> 11
- Input: 224x224 (resized to match DenseNet)
- Same coordinate-to-measurement geometry as cnn_on_points (`common/geometry.py`)

Requires `torchxrayvision` (`pip install torchxrayvision`); the DenseNet weights download on first use.

## Usage
```bash
# train (run from project root)
python3 -m pretrained_cnn.train --mdata 20260407.csv --idata box_images --esl 5

# test
python3 -m pretrained_cnn.test --mdata 20260407.csv --idata box_images --ds test
```

`EXPERIMENTATION/xrv_autoencoder.py` is a standalone scratch script; run it from the project root as well (`python3 pretrained_cnn/EXPERIMENTATION/xrv_autoencoder.py`).
