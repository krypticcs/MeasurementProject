#run from project root: python -m experiments.summarize [--mdata 20260427.csv] [--only name1 name2 ...]
#reads every experiments/results/<name>.csv (copies of a test script's results.csv) and prints markdown tables
#of per-measurement mean percent error, mean absolute error and std of absolute errors, plus a training-mean
#null model evaluated on the exact same seed-42 test split
import os
import argparse
import numpy as np
import pandas as pd

from common.geometry import MEASUREMENT_NAMES
from common.data import MeasurementSplit

RESULTS_DIR = "./experiments/results"

#per-side arrays (n, 10) of predictions and truths from a results.csv
def load_results(path):
    df = pd.read_csv(path)
    df = df[df["id"].astype(str) != "avg"]
    pred = np.stack([df[f"{m} (pred)"].to_numpy(dtype = float) for m in MEASUREMENT_NAMES], axis = 1)
    true = np.stack([df[f"{m} (true)"].to_numpy(dtype = float) for m in MEASUREMENT_NAMES], axis = 1)
    return pred, true

def stats(pred, true):
    abs_err = np.abs(pred - true)
    return {"pe": (abs_err / true).mean(axis = 0), "mae": abs_err.mean(axis = 0), "std": abs_err.std(axis = 0, ddof = 1), "n": len(true)}

def null_model(mdata, idata, seed):
    split = MeasurementSplit(mdata, idata, (0.8, 0.1), seed)
    side = lambda files: np.stack([split.side_y(f, s).numpy() for f in files for s in range(2)])
    mean = side(split.splits["train"]).mean(axis = 0)
    true = side(split.splits["test"])
    return stats(np.broadcast_to(mean, true.shape), true)

def table(rows, key, fmt):
    short = ["M.Cort", "L.Cort", "Shaft", "Head", "H.Off", "V.Off", "Neck W", "HAL", "NAL", "NSA"]
    out = ["| Model | n | " + " | ".join(short) + " | Mean |", "|---|---|" + "---|" * (len(short) + 1)]
    for name, s in rows:
        vals = s[key]
        out.append(f"| {name} | {s['n']} | " + " | ".join(fmt(v) for v in vals) + f" | {fmt(vals.mean())} |")
    return "\n".join(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Summarize test results")
    parser.add_argument("--mdata", type = str, default = "20260427.csv", required = False, help = "Measurement file used for the split (null model)")
    parser.add_argument("--idata", type = str, default = "box_images", required = False, help = "Image directory (from ./data)")
    parser.add_argument("--seed", type = int, default = 42, required = False, help = "Split seed")
    parser.add_argument("--only", type = str, nargs = "*", default = None, required = False, help = "Only these result names, in this order")
    args = parser.parse_args()

    rows = [("null (train mean, same split)", null_model(args.mdata, args.idata, args.seed))]
    names = args.only if args.only else sorted(f[:-4] for f in os.listdir(RESULTS_DIR) if f.endswith(".csv"))
    for name in names:
        rows.append((name, stats(*load_results(os.path.join(RESULTS_DIR, name + ".csv")))))

    print("\n### Mean percent error (|pred - true| / true)\n")
    print(table(rows, "pe", lambda v: f"{100 * v:.2f}%"))
    print("\n### Mean absolute error (mm, degrees for NSA)\n")
    print(table(rows, "mae", lambda v: f"{v:.3f}"))
    print("\n### Std dev of absolute error\n")
    print(table(rows, "std", lambda v: f"{v:.3f}"))
