# cnn_backbone

CNN measurement model that uses a pretrained autoencoder encoder as its backbone, with a head that regresses 11 keypoint parameters.

## Architecture
- **Backbone**: Encoder from `autoencoder/` (channels `[1, 64, 128, 32, 1]`), loaded from `autoencoder/model_saves/autoencoder.pth`
- **Head**: Conv + linear layers on top of encoded features -> 11 keypoint parameters
- Same coordinate-to-measurement geometry as cnn_on_points (`common/geometry.py`)

By default the backbone is fine-tuned together with the head. Pass `--freeze_enc` to freeze the encoder and only train the head.

Requires `autoencoder/model_saves/autoencoder.pth` (see "Required weights" in the root README).

## Usage
```bash
# train (run from project root)
python3 -m cnn_backbone.train --mdata 20260407.csv --idata box_images --esl 5

# train with frozen encoder
python3 -m cnn_backbone.train --mdata 20260407.csv --idata box_images --esl 5 --freeze_enc

# test
python3 -m cnn_backbone.test --mdata 20260407.csv --idata box_images --ds test
```
