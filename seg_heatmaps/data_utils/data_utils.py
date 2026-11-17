#thin wrapper around the shared loader in common/data.py: every image is cached once at 2x resolution, the training set gets
#fresh random augmentation on every draw (ranges passed to create_ds are (min, max, step) bounds), validation/test are never augmented
#always reads ./data/box_images (also used by dual_heatmaps)
from common.data import DataProcessor as SharedDataProcessor

class DataProcessor(SharedDataProcessor):
    def __init__(self, measurement_file, train_val_split, batch_size, img_width, img_height, seed):
        super().__init__(measurement_file, "box_images", train_val_split, batch_size, img_width, img_height, seed)
