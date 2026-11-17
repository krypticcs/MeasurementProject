#native DXA resolution is 2400 x 1920 (408 mm wide); model inputs are a uniform downscale of that

NATIVE_WIDTH = 2400
NATIVE_HEIGHT = 1920

def image_size(scale):
    """Return (width, height) at the given scale factor of native DXA resolution."""
    return int(NATIVE_WIDTH * scale), int(NATIVE_HEIGHT * scale)

def default_batch_sizes(scale, train_bs, val_bs, test_bs, train_default = 64, val_default = 64, test_default = 32):
    """At >= 0.2x, cut the usual 0.1x batch-size defaults (~4x pixels) unless the user overrode them."""
    if scale < 0.2:
        return train_bs, val_bs, test_bs
    if train_bs == train_default:
        train_bs = 16
    if val_bs == val_default:
        val_bs = 16
    if test_bs == test_default:
        test_bs = 8
    return train_bs, val_bs, test_bs
