#imports
import torch
import torch.nn.functional as F

#refer to powerpoint: https://docs.google.com/presentation/d/1FPTIKSnscUQzuTzK1f5JnABffidgGuaxPcGEeSBpgRY/edit?slide=id.g362ec749174_0_26#slide=id.g362ec749174_0_26

#original DXA images are 2400 pixels wide, representing 408mm
PIX_PER_MM = 2400 / 408

POINTS_LABELS = ['D', 'F', 'G', 'H', 'W', 'X', 'Y', 'Z', 'H\'', 'A\'', 'A\'\'', 'C']
HEATMAP_LABELS = ['D', 'F', 'G', 'H', 'W', 'X', 'Y', 'Z', 'H\'', 'A\'', 'A\'\'', 'B']
HEATMAP_ABS_LABELS = ['A'] + HEATMAP_LABELS

MEASUREMENT_NAMES = ["M. Cortical", "L. Cortical", "Shaft Width", "F. Head Diam.", "H. Offset", "V. Offset", "F. Neck Width", "Hip Axis Length", "F. Neck Axis Length", "F. Neck-Shaft Ang."]

#####direct regression models (11 params -> 12 points, last point is C)#####

#turn model output into (coordinate, length ab)
def model_to_coord(model_out):
    #units: pixels -> pixels
    #shape: (batch_size, 11)

    eps = 1e-6

    slope = model_out[:, 1] / (model_out[:, 0] + eps)

    all_x_y = [None for _ in range(12)]
    all_x_y[0] = torch.stack([model_out[:, 0], model_out[:, 1]], dim = -1) #D
    all_x_y[1] = torch.stack([model_out[:, 2], slope * model_out[:, 2]], dim = -1) #F
    all_x_y[2] = torch.stack([model_out[:, 3], slope * model_out[:, 3]], dim = -1) #G
    all_x_y[3] = torch.stack([model_out[:, 4], model_out[:, 5]], dim = -1) #H
    all_x_y[4] = torch.stack([model_out[:, 6], model_out[:, 7]], dim = -1) #W
    all_x_y[5] = torch.stack([model_out[:, 8], model_out[:, 7]], dim = -1) #X
    all_x_y[6] = torch.stack([model_out[:, 9], model_out[:, 7]], dim = -1) #Y
    all_x_y[7] = torch.stack([model_out[:, 10], model_out[:, 7]], dim = -1) #Z

    b_x = (model_out[:, 4] / (slope + eps) + model_out[:, 5]) / (slope + 1 / (slope + eps) + eps)
    b_y = slope * b_x
    ab = torch.sqrt(b_x ** 2 + b_y ** 2)

    all_x_y[8] = torch.stack([2 * b_x - model_out[:, 4], 2 * b_y - model_out[:, 5]], dim = -1) #H'
    all_x_y[9] = torch.stack([model_out[:, 1], -model_out[:, 0]], dim = -1) #A'
    all_x_y[10] = torch.stack([-model_out[:, 1], model_out[:, 0]], dim = -1) #A''
    all_x_y[11] = torch.stack([(model_out[:, 6] + model_out[:, 10]) / 2, model_out[:, 7]], dim = -1) #C

    coord = torch.stack(all_x_y, dim = 1)

    return coord, ab

#####heatmap models (13 absolute points -> 12 relative points, last point is B)#####

#rotate points (around origin) so that W -> Z is horizontal
def rot_coord(coord):
    #W: 4
    #Z: 7

    B = coord.shape[0]
    K = coord.shape[1]

    vec = coord[:, 7, :] - coord[:, 4, :]
    ang = torch.atan2(vec[:, 1], vec[:, 0])
    sinang = torch.sin(ang)
    cosang = torch.cos(ang)
    rotmat = torch.stack([
        torch.stack([cosang, -sinang], dim = -1),
        torch.stack([sinang, cosang], dim = -1)
    ], dim = -1).view(B, 1, 2, 2)
    res = rotmat @ coord.view(B, K, 2, 1)

    return res.view(B, K, 2)

#translate and rotate absolute coordinates to relative coordinates
def abs_kp_to_coord(abs_kp):
    #abs_kp shape: (batch_size, 13, 2), A first

    #translate
    trans_kp = abs_kp[:, 1:, :] - abs_kp[:, 0:1, :]

    #rotate
    trans_kp = rot_coord(trans_kp)

    return trans_kp

#get ab from model relative coord
def get_ab(coord):
    #B: 11
    ab = torch.norm(coord[:, 11], dim = 1)
    return ab

#heatmaps to keypoints (soft-argmax), in pixels of an img_height x img_width image (defaults to heatmap size)
def heatmaps_to_keypoints(heatmaps, img_height = None, img_width = None):
    device = heatmaps.device

    B, K, H, W = heatmaps.shape
    if img_height is None:
        img_height = H
    if img_width is None:
        img_width = W

    heatmaps = heatmaps.view(B, K, -1)
    softmax_heatmaps = F.softmax(heatmaps, dim = -1)
    softmax_heatmaps = softmax_heatmaps.view(B, K, H, W)

    xs = torch.linspace(0, img_width, W, device = device)
    ys = torch.linspace(0, img_height, H, device = device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing = 'ij')
    grid_x = grid_x.unsqueeze(0).unsqueeze(0)
    grid_y = grid_y.unsqueeze(0).unsqueeze(0)

    expected_x = (softmax_heatmaps * grid_x).sum(dim = [-2, -1])
    expected_y = (softmax_heatmaps * grid_y).sum(dim = [-2, -1])

    return torch.stack([expected_x, expected_y], dim = 2)

#####shared#####

#points 0-10 (D, F, G, H, W, X, Y, Z, H', A', A'') plus b and c
def _measurements_to_base_points(measurements, ab, pix_per_mm, img_scale_factor):
    #units: mm -> pixels
    #measurements shape: (batch_size, 10)
    #ab shape: (batch_size,)

    measurements = measurements.clone()
    measurements[:, :9] *= pix_per_mm * img_scale_factor

    ang = (270 - measurements[:, 9]) * torch.pi / 180
    cos = torch.cos(ang)
    sin = torch.sin(ang)

    b_x = -cos * ab
    b_y = -sin * ab
    c_x = measurements[:, 4]
    c_y = -measurements[:, 5]

    all_x_y = [None for _ in range(11)]
    all_x_y[0] = torch.stack([cos * measurements[:, 3] / 2, sin * measurements[:, 3] / 2], dim = -1) #D
    all_x_y[2] = torch.stack([all_x_y[0][:, 0] - cos * measurements[:, 8], all_x_y[0][:, 1] - sin * measurements[:, 8]], dim = -1) #G
    all_x_y[1] = torch.stack([all_x_y[2][:, 0] + cos * measurements[:, 7], all_x_y[2][:, 1] + sin * measurements[:, 7]], dim = -1) #F
    all_x_y[3] = torch.stack([b_x + sin * measurements[:, 6] / 2, b_y - cos * measurements[:, 6] / 2], dim = -1) #H
    all_x_y[4] = torch.stack([c_x - measurements[:, 2] / 2, c_y], dim = -1) #W
    all_x_y[7] = torch.stack([c_x + measurements[:, 2] / 2, c_y], dim = -1) #Z
    all_x_y[5] = torch.stack([all_x_y[4][:, 0] + measurements[:, 0], c_y], dim = -1) #X
    all_x_y[6] = torch.stack([all_x_y[7][:, 0] - measurements[:, 1], c_y], dim = -1) #Y

    all_x_y[8] = torch.stack([b_x - sin * measurements[:, 6] / 2, b_y + cos * measurements[:, 6] / 2], dim = -1) #H'
    all_x_y[9] = torch.stack([sin * measurements[:, 3] / 2, -cos * measurements[:, 3] / 2], dim = -1) #A'
    all_x_y[10] = torch.stack([-sin * measurements[:, 3] / 2, cos * measurements[:, 3] / 2], dim = -1) #A''

    return all_x_y, (b_x, b_y), (c_x, c_y)

#turn measurements into points for direct regression models (last point is C, y up)
def measurements_to_coord_points(measurements, ab, pix_per_mm, img_scale_factor):
    all_x_y, _, (c_x, c_y) = _measurements_to_base_points(measurements, ab, pix_per_mm, img_scale_factor)
    all_x_y.append(torch.stack([c_x, c_y], dim = -1)) #C
    return torch.stack(all_x_y, dim = 1)

#turn measurements into points for heatmap models (last point is B, y down like image rows)
def measurements_to_coord_heatmap(measurements, ab, pix_per_mm, img_scale_factor):
    all_x_y, (b_x, b_y), _ = _measurements_to_base_points(measurements, ab, pix_per_mm, img_scale_factor)
    all_x_y.append(torch.stack([b_x, b_y], dim = -1)) #B
    coord = torch.stack(all_x_y, dim = 1)
    coord[:, :, 1] *= -1
    return coord

#turn points into measurements (flip_y = True for heatmap model coordinates)
def coord_to_measurements(coord, pix_per_mm, img_scale_factor, flip_y = False):
    #units: pixels -> mm
    #shape (batch_size, 12, 2)

    if flip_y:
        coord = coord.clone()
        coord[:, :, 1] *= -1

    eps = 1e-6

    slope = coord[:, 0, 1] / (coord[:, 0, 0] + eps)
    b_x = (coord[:, 3, 0] / (slope + eps) + coord[:, 3, 1]) / (slope + 1 / (slope + eps) + eps)
    b_y = slope * b_x

    measurements = [None for _ in range(10)]
    measurements[0] = coord[:, 5, 0] - coord[:, 4, 0]
    measurements[1] = coord[:, 7, 0] - coord[:, 6, 0]
    measurements[2] = coord[:, 7, 0] - coord[:, 4, 0]
    measurements[3] = 2 * torch.norm(coord[:, 0], dim = 1)
    measurements[4] = (coord[:, 4, 0] + coord[:, 7, 0]) / 2
    measurements[5] = -coord[:, 4, 1]
    measurements[6] = 2 * torch.sqrt((coord[:, 3, 0] - b_x) ** 2 + (coord[:, 3, 1] - b_y) ** 2)
    measurements[7] = torch.norm(coord[:, 1] - coord[:, 2], dim = 1)
    measurements[8] = torch.norm(coord[:, 0] - coord[:, 2], dim = 1)
    measurements[9] = torch.arctan(-slope) * 180 / torch.pi + 90

    measurements = torch.stack(measurements, dim = -1).clone()
    measurements[:, :9] /= pix_per_mm * img_scale_factor

    return measurements
