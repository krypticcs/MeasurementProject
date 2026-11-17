#imports
import numpy as np
from scipy import ndimage
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from common.cache_images import load_half, HALF_HEIGHT, HALF_WIDTH
from common.data import resize
from common.geometry import PIX_PER_MM, abs_kp_to_coord, coord_to_measurements, heatmaps_to_keypoints
from SEGMENTATIONS.mod_seg.model.model import SegUNet

#stage 1 runs on the 0.1x image, stage 2 crops from the 0.5x image
S1_WIDTH = 240
S1_HEIGHT = 192
S1_SCALE = 0.1
S1_PX_PER_MM = PIX_PER_MM * S1_SCALE
HALF_PX_PER_MM = PIX_PER_MM * 0.5

#half-res patch kept per (image, side) around the shaft; must contain the largest jittered crop
PATCH = 512
CROP_H = 64
CROP_W = 256
MIN_WIDTH = 40
MAX_WIDTH = 220

#shaft rows used for the first axis fit (0.1x pixels above the lowest femur row)
SHAFT_ROWS = 20
BOTTOM_MARGIN = 3

def load_stage1(path, device):
    model = SegUNet(13, final_act = False)
    model.load_state_dict(torch.load(f"./seg_heatmaps/model_saves/{path}", map_location = "cpu"))
    return model.to(device).eval()

def load_segmenter(device, path = "./SEGMENTATIONS/mod_seg/model_saves/segunet.pth"):
    model = SegUNet(1)
    model.load_state_dict(torch.load(path, map_location = "cpu"))
    return model.to(device).eval()

#0.1x pixel index coordinates -> half-res pixel index coordinates (0.1x pixel j covers half-res pixels 5j..5j+4)
def s1_to_half(xy):
    f = HALF_WIDTH / S1_WIDTH
    return xy * f + (f - 1) / 2

#least squares x = a * y + b through the mask's row midpoints over rows [y0, y1]
def fit_axis(mask, y0, y1):
    rows, mids = [], []
    for y in range(max(y0, 0), min(y1, mask.shape[0] - 1) + 1):
        xs = np.nonzero(mask[y])[0]
        if len(xs):
            rows.append(y)
            mids.append((xs[0] + xs[-1]) / 2)
    if len(rows) < 3:
        return None
    return np.polyfit(rows, mids, 1)

#anchor the shaft line in absolute 0.1x pixel coordinates from the femur mask
#the heatmap models are only trained on keypoints relative to A (head center), so their absolute positions are not
#tied to the anatomy; the mask is. Head center = deepest mask point medial to the shaft axis within one head diameter
#of the top of the femur (lower down, the intertrochanteric region is thicker than the head), shaft axis = line through
#the mask's row midpoints, and the W-Z line center sits on the axis vo_px below the head center, like V. offset is defined
#returns (center (2,), theta = angle of the medial -> lateral direction W -> Z, ok, head center (2,))
def localize_shaft(prob, vo_px, hd_px):
    mask = prob > 0.5
    lab, n = ndimage.label(mask)
    if n == 0:
        return np.array([0.8 * S1_WIDTH, 0.9 * S1_HEIGHT]), 0.0, False, None
    mask = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    ys = np.nonzero(mask.any(axis = 1))[0]
    y_top, y_bot = ys[0], ys[-1]

    axis = fit_axis(mask, y_bot - SHAFT_ROWS, y_bot)
    if axis is None:
        return np.array([mask.nonzero()[1].mean(), y_bot]), 0.0, False, None

    dist = ndimage.distance_transform_edt(mask)
    yy, xx = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
    head_region = mask & (yy < min(y_top + hd_px, y_bot - SHAFT_ROWS)) & (xx < axis[0] * yy + axis[1])
    if not head_region.any():
        return np.array([axis[0] * y_bot + axis[1], y_bot]), 0.0, False, None
    hy, hx = np.unravel_index(np.argmax(np.where(head_region, dist, -1)), mask.shape)

    refined = fit_axis(mask, int(hy + 0.6 * vo_px), y_bot)
    if refined is not None:
        axis = refined

    d = np.array([axis[0], 1.0]) / np.hypot(axis[0], 1.0) #axis direction, pointing down the shaft
    p0 = np.array([axis[1], 0.0])
    head = np.array([hx, hy], dtype = float)
    foot = p0 + np.dot(head - p0, d) * d
    center = foot + vo_px * d
    if center[1] > y_bot - BOTTOM_MARGIN: #the line has to be on the visible femur
        center = center - (center[1] - (y_bot - BOTTOM_MARGIN)) / d[1] * d
    theta = np.arctan2(-d[0], d[1]) #perpendicular to the axis, pointing to +x (lateral)
    return center, theta, True, head

#stage-1 predictions and a PATCH x PATCH half-res patch around the shaft line for every (image, side) of a split
#side 0 = unflipped image / left measurements, side 1 = flipped image / right measurements (same as every DataProcessor);
#in both cases the femur being measured is on the right of the (flipped) image, which is the one the segmenter outlines
class ShaftSet:
    def __init__(self, split, ds_name, stage1, segmenter, device):
        self.items = [(f, s) for f in split.splits[ds_name] for s in range(2)]
        patches, centers, thetas, widths, s1_meas, ys, oks = [], [], [], [], [], [], []

        with torch.no_grad():
            for img_file in tqdm(split.splits[ds_name], unit = "image"):
                half = load_half(img_file, split.img_dir)
                half = torch.cat((half, TF.hflip(half)), dim = 0) #(2, 1, H, W): side 0, side 1
                low = resize(half, S1_HEIGHT, S1_WIDTH).to(device)

                meas = coord_to_measurements(abs_kp_to_coord(heatmaps_to_keypoints(stage1(low))), PIX_PER_MM, S1_SCALE, flip_y = True).cpu()
                prob = segmenter(low).cpu().numpy()[:, 0]
                s1_meas.append(meas)

                padded = F.pad(half, (PATCH, PATCH, PATCH, PATCH))
                for side in range(2):
                    center, theta, ok, _ = localize_shaft(prob[side], meas[side, 5].item() * S1_PX_PER_MM, meas[side, 3].item() * S1_PX_PER_MM)
                    center = s1_to_half(torch.tensor(center, dtype = torch.float32))
                    ox = min(max(int(round(center[0].item())) - PATCH // 2, -PATCH), HALF_WIDTH)
                    oy = min(max(int(round(center[1].item())) - PATCH // 2, -PATCH), HALF_HEIGHT)
                    patches.append(padded[side, :, oy + PATCH:oy + 2 * PATCH, ox + PATCH:ox + 2 * PATCH].half())
                    centers.append(center - torch.tensor([ox, oy], dtype = torch.float32))
                    thetas.append(float(theta))
                    widths.append(min(max(meas[side, 2].item() * HALF_PX_PER_MM, MIN_WIDTH), MAX_WIDTH))
                    oks.append(ok)
                    ys.append(split.side_y(img_file, side))

        self.patches = torch.stack(patches) #(N, 1, PATCH, PATCH) float16
        self.centers = torch.stack(centers) #(N, 2) shaft line center in patch pixel coordinates
        self.thetas = torch.tensor(thetas) #(N,) angle of the W -> Z direction
        self.widths = torch.tensor(widths) #(N,) stage-1 shaft width in half-res pixels
        self.s1_meas = torch.cat(s1_meas) #(N, 10) stage-1 measurements
        self.y = torch.stack(ys) #(N, 10) true measurements
        self.localized = torch.tensor(oks)
        print(f"Loaded {ds_name} with {len(self.items)} points ({(~self.localized).sum().item()} without a usable femur mask)")

    def __len__(self):
        return len(self.items)

    def batch(self, idx, device):
        return (self.patches[idx].to(device).float(), self.centers[idx].to(device), self.thetas[idx].to(device),
                self.widths[idx].to(device), self.y[idx].to(device))

#rotated crop: columns run along stage-1 W -> Z, CROP_W columns span 2x the stage-1 shaft width, rows use the same pixel scale
#jitter (training) perturbs center, angle and scale to simulate stage-1 error
#returns standardized crops (B, 1, CROP_H, CROP_W) and s = half-res pixels per crop column
def make_crops(patches, centers, thetas, widths, jitter = False):
    B = patches.shape[0]
    dev = patches.device
    scale = torch.ones(B, device = dev)
    du = torch.zeros(B, device = dev)
    dv = torch.zeros(B, device = dev)
    if jitter:
        scale = scale.uniform_(0.85, 1.15)
        thetas = thetas + torch.empty(B, device = dev).uniform_(-5, 5) * np.pi / 180
        du = torch.empty(B, device = dev).uniform_(-0.1, 0.1) * widths
        dv = torch.empty(B, device = dev).uniform_(-16, 16)

    s = 2 * widths * scale / CROP_W
    cos, sin = torch.cos(thetas), torch.sin(thetas)
    cx = centers[:, 0] + du * cos - dv * sin
    cy = centers[:, 1] + du * sin + dv * cos

    u = (torch.arange(CROP_W, device = dev) - (CROP_W - 1) / 2).view(1, 1, CROP_W) * s.view(B, 1, 1)
    v = (torch.arange(CROP_H, device = dev) - (CROP_H - 1) / 2).view(1, CROP_H, 1) * s.view(B, 1, 1)
    px = cx.view(B, 1, 1) + u * cos.view(B, 1, 1) - v * sin.view(B, 1, 1)
    py = cy.view(B, 1, 1) + u * sin.view(B, 1, 1) + v * cos.view(B, 1, 1)
    P = patches.shape[-1]
    grid = torch.stack(((2 * px + 1) / P - 1, (2 * py + 1) / P - 1), dim = -1)

    crops = F.grid_sample(patches, grid, mode = "bilinear", padding_mode = "zeros", align_corners = False)
    mean = crops.mean(dim = (1, 2, 3), keepdim = True)
    std = crops.std(dim = (1, 2, 3), keepdim = True) + 1e-5
    return (crops - mean) / std, s

#crop-column positions of W, X, Y, Z -> [M. cortical (X - W), L. cortical (Z - Y), shaft width (Z - W)] in mm
def positions_to_measurements(pos, s):
    mm = pos * s.unsqueeze(1) / HALF_PX_PER_MM
    return torch.stack((mm[:, 1] - mm[:, 0], mm[:, 3] - mm[:, 2], mm[:, 3] - mm[:, 0]), dim = 1)
