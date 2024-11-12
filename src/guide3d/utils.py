from typing import Dict, List

import numpy as np
from scipy.interpolate import splev


def decompose_path(img_path: str) -> str:
    path = img_path.split("/")[0]
    camera = path.split("-")[-2]
    img_number = img_path.split("/")[-1].split(".")[0]
    return path, camera, img_number


def preprocess_tck(tck: Dict) -> List:
    t, c, k = tck["t"], tck["c"], tck["k"]

    t = np.array(t)
    c = np.array([np.array(c_i) for c_i in c]).T
    c = np.clip(c, 0, 1024)
    k = int(k)

    return t, c, k


def sample_spline(tck: tuple, u: list = None, n: int = None, delta: float = None):
    assert delta or n, "Either delta or n must be provided"
    assert not (delta and n), "Only one of delta or n must be provided"

    def is2d(tck):
        return len(tck[1]) == 2

    u_max = u[-1] if u is not None else tck[0][-1]
    num_samples = int(u_max / delta) + 1 if delta else n
    u = np.linspace(0, u_max, num_samples)
    if is2d(tck):
        x, y = splev(u, tck, ext=3)
        return np.column_stack([x, y]).astype(np.int32)
    else:
        x, y, z = splev(u, tck)
        return np.column_stack([x, y, z])


def split_fn_image(
    data: List,
    split: tuple = (0.8, 0.1, 0.1),
) -> List:
    train_data = []
    val_data = []
    test_data = []

    for video in data:
        train_idx = int(split[0] * len(video))
        val_idx = int(split[1] * len(video))
        train_data.extend(video[:train_idx])
        val_data.extend(video[train_idx : train_idx + val_idx])
        test_data.extend(video[train_idx + val_idx :])
    return train_data, val_data, test_data


def split_fn_video(
    data: List,
    split: tuple = (0.8, 0.1, 0.1),
) -> List:
    train_data = []
    val_data = []
    test_data = []

    for video in data:
        train_idx = int(split[0] * len(video))
        val_idx = int(split[1] * len(video))
        train_data.extend(video[:train_idx])
        val_data.extend(video[train_idx : train_idx + val_idx])
        test_data.extend(video[train_idx + val_idx :])
    return train_data, val_data, test_data
