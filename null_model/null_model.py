#imports
import argparse
import numpy as np
import pandas as pd

#parser
parser = argparse.ArgumentParser(description = "Null model (training set mean) baseline")

parser.add_argument("--mdata", type = str, default = "20260427.csv", required = False, help = "Path to measurement data file (from ./data/measurements)")
parser.add_argument("--seed", type = int, default = 42, required = False, help = "Numpy seed for the train/test split")
parser.add_argument("--train_split", type = float, default = 0.8, required = False, help = "Training set split")

args = parser.parse_args()

#settings
measurement_file = args.mdata
train_test_split = args.train_split
np.random.seed(args.seed)

#read csv
df = pd.read_csv(f"./data/measurements/{measurement_file}")
df.drop(columns = ["ID", "Student Name", "Notes", "Date Completed"], inplace = True, errors = "ignore")
df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
df = df.apply(pd.to_numeric, errors = "coerce").dropna()

#train test split
all_idx = np.random.permutation(len(df))
train_sz = int(len(df) * train_test_split)
train_idx = all_idx[:train_sz]
test_idx = all_idx[train_sz:]

#model
model = np.zeros(10)

#training
for i in train_idx:
    model += df.iloc[i][:10].to_numpy()
    model += df.iloc[i][10:].to_numpy()
model /= 2 * len(train_idx)

print(model)

#testing
err = np.zeros(10)
for i in test_idx:
    lside = df.iloc[i][:10].to_numpy()
    rside = df.iloc[i][10:].to_numpy()
    err += np.abs((lside - model) / lside)
    err += np.abs((rside - model) / rside)
err /= 2 * len(test_idx)
for i in err:
    print(f'{i.item():.4f}', end = ' ')
print()
