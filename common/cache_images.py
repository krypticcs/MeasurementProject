#run from project root: python -m common.cache_images --mdata 20260427.csv [--idata box_images]
#writes every image used by the measurement file as float16 .npy at 0.5x scale (960x1200, 2.94 px/mm) to data/cache/half/
#images are min-max normalized at full resolution first (same as every DataPoint) and stored unflipped
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch

from common.data import CACHE_ROOT, load_dicom, read_measurements, resize

HALF_DIR = os.path.join(CACHE_ROOT, "half")
HALF_HEIGHT = 960
HALF_WIDTH = 1200

def half_path(img_file):
    return os.path.join(HALF_DIR, img_file + ".npy")

#(1, 1, 960, 1200) float32 tensor, built from the DICOM and cached on first use
#a 2x bilinear (align_corners = False) downscale is an exact 2x2 block average, so resizing this by another 5x
#gives the same 192x240 image the 0.1x models are trained on
def load_half(img_file, img_dir = "./data/box_images"):
    path = half_path(img_file)
    if os.path.exists(path):
        return torch.from_numpy(np.load(path)).float()
    half = resize(load_dicom(os.path.join(img_dir, img_file)), HALF_HEIGHT, HALF_WIDTH).half()
    os.makedirs(HALF_DIR, exist_ok = True)
    np.save(path, half.numpy())
    return half.float()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Cache images at 0.5x scale")
    parser.add_argument("--mdata", type = str, required = True, help = "Measurement file (from ./data/measurements); only its complete rows are cached")
    parser.add_argument("--idata", type = str, default = "box_images", required = False, help = "Image directory (from ./data)")
    args = parser.parse_args()

    img_dir = "./data/" + args.idata
    files = sorted(set(read_measurements(args.mdata)["ID"].values) & set(os.listdir(img_dir)))
    todo = [f for f in files if not os.path.exists(half_path(f))]
    print(f"{len(files)} images, {len(files) - len(todo)} already cached, caching {len(todo)} to {HALF_DIR}")
    for img_file in tqdm(todo, unit = "image"):
        load_half(img_file, img_dir)
