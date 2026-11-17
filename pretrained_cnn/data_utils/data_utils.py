#thin wrapper around the shared loader in common/data.py: every image is cached once at 2x resolution, the training set gets
#fresh random augmentation on every draw (ranges passed to create_ds are (min, max, step) bounds), validation/test are never augmented
#images are zero-padded top/bottom to a sq_dim x sq_dim square for the DenseNet backbone
from common.data import DataProcessor as SharedDataProcessor

class DataProcessor(SharedDataProcessor):
    def __init__(self, measurement_file, img_dir, train_val_split, batch_size, img_width, img_height, sq_dim, seed):
        super().__init__(measurement_file, img_dir, train_val_split, batch_size, img_width, img_height, seed, sq_dim = sq_dim)
