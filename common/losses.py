#imports
import torch

from common.geometry import coord_to_measurements

#mean relative (percent) error of the 10 measurements implied by predicted keypoints
#yvals are already rescaled for augmentation by the data loaders, so this is scale invariant
def measurement_loss(model_coord, yvals, pix_per_mm, img_scale_factor, flip_y = False, eps = 1e-3):
    ypred = coord_to_measurements(model_coord, pix_per_mm, img_scale_factor, flip_y = flip_y)
    return (torch.abs(ypred - yvals) / (yvals.abs() + eps)).mean()
