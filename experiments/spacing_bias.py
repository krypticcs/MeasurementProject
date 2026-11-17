#run from project root: python -m experiments.spacing_bias <result name> [<result name> ...]
#splits a test results.csv by the DICOM PixelSpacing of each image; common/geometry.py assumes 0.17 mm/px (2400 px = 408 mm)
#for every image, so if the labels are in calibrated mm, images at 0.15 mm/px should come out ~13% too large
#rows of every test results.csv are (test image, side) in split order, side 0 then side 1
import sys
import warnings
import numpy as np
import pydicom

from common.data import MeasurementSplit
from experiments.summarize import load_results

warnings.filterwarnings("ignore")
split = MeasurementSplit("20260427.csv", "box_images", (0.8, 0.1), 42)
spacing = np.array([float(pydicom.dcmread(f"./data/box_images/{f}", stop_before_pixels = True).PixelSpacing[0]) for f in split.splits["test"] for _ in range(2)])
groups = {"0.150 mm/px": spacing < 0.16, ">= 0.169 mm/px": spacing >= 0.16}

for name in sys.argv[1:]:
    pred, true = load_results(f"./experiments/results/{name}.csv")
    print(f"\n{name}: pred / true on the 9 length measurements (median), and mean percent error over all 10")
    for g, m in groups.items():
        ratio = np.median(pred[m, :9] / true[m, :9])
        pe = np.mean(np.abs(pred[m] - true[m]) / true[m])
        print(f"  {g:>15}: n = {m.sum():3d}  pred/true {ratio:.3f}  percent error {100 * pe:.2f}%")
