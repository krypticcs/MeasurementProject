# null_model

Baseline model that predicts the training set mean for all 10 measurements. No neural network -- just computes the average of each measurement across the training set (left and right sides pooled) and uses that as the prediction for every test sample. Prints the mean vector and the mean percent error per measurement.

## Usage
```bash
# run from project root (defaults to the newest measurement file)
python3 -m null_model.null_model

# choose a measurement file / split seed
python3 -m null_model.null_model --mdata 20251014.csv --seed 0
```
