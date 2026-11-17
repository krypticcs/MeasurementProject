#thin wrapper around the shared loader in common/data.py: every image is cached once at 2x resolution, the training set gets
#fresh random augmentation on every draw (ranges passed to create_ds are (min, max, step) bounds), validation/test are never augmented
from common.data import DataProcessor
