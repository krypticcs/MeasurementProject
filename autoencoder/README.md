# autoencoder

U-Net autoencoder for DXA image reconstruction. Encoder-decoder with optional skip connections (residual mode). Used as a backbone for other models (e.g., cnn_backbone).

## Architecture
- **Encoder**: stride-2 downsampling, BatchNorm, ReLU; channel counts set by `channel_cnt` (train/test use `[1, 64, 128, 32, 1]`)
- **Decoder**: Mirrors encoder with bilinear upsampling, optional skip connections from encoder
- Sigmoid output for image reconstruction
- Input: 192x240 grayscale

## Usage
```bash
# train (run from project root)
python3 -m autoencoder.train --idata box_images --esl 5

# test
python3 -m autoencoder.test --idata box_images --ds test
```

Weights are saved to `autoencoder/model_saves/` (`current_best.pth`, `final_model.pth`). `cnn_backbone` loads `autoencoder/model_saves/autoencoder.pth`, so copy the run you want to use there (see "Required weights" in the root README).
