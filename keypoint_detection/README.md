# keypoint_detection

Self-supervised keypoint detection using image reconstruction. A location encoder produces keypoint heatmaps from a target image, an image encoder encodes a source image, and a decoder reconstructs the source from the target's keypoints + source's encoding.

## Architecture
- **ImageEncoder**: Encodes source image to feature map
- **LocationEncoder**: Produces K keypoint heatmaps from target image, converted to Gaussian maps via soft-argmax
- **Decoder**: Reconstructs source image from concatenated source features + target Gaussian keypoints
- Learns keypoints without explicit coordinate supervision

## Usage
```bash
# train (run from project root)
python3 -m keypoint_detection.train --idata box_images --esl 5 --loss vgg

# test
python3 -m keypoint_detection.test --idata box_images --ds test
```
