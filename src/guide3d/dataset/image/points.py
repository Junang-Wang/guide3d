from pathlib import Path
from typing import Dict, List, Union

import guide3d.representations.curve as curve
import numpy as np
import torch
import torch.nn.functional as F
from guide3d.dataset.dataset_utils import BaseGuide3D
from guide3d.utils.utils import preprocess_tck
from torch.utils import data
from torchvision import transforms
from torchvision.io import read_image

IMAGE_SIZE = 1024
N_CHANNELS = 1
MODEL_VERSION = "1"

image_transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert image to PIL image
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize image to 224x224
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(N_CHANNELS, 1, 1)),
        transforms.Normalize(  # Normalize with mean and std
            mean=[0.5 for _ in range(N_CHANNELS)],
            std=[0.5 for _ in range(N_CHANNELS)],
        ),
    ]
)


def c_transform(c):
    return c / IMAGE_SIZE


def t_transform(t):
    return t / 2000


def c_untransform(c):
    return c * IMAGE_SIZE


def t_untransform(t):
    return t * 2000


def unnorm(img):
    img = img * 0.5 + 0.5
    return img


def filter_pts(pts):
    filtered = []
    for pt in pts:
        if 0 <= pt[0] < 1024 and 0 <= pt[1] < 1024:
            filtered.append(pt)
    return np.array(filtered)


def process_data(
    data: Dict,
) -> List:
    video_pairs = []
    for video_pair in data:
        videoA = []
        videoB = []
        for frame in video_pair["frames"]:
            imageA = frame["cameraA"]["image"]
            imageB = frame["cameraB"]["image"]

            tckA = preprocess_tck(frame["cameraA"]["tck"])
            tckB = preprocess_tck(frame["cameraB"]["tck"])

            videoA.append(dict(image=imageA, pts=curve.sample_spline((tckA[0], tckA[1].T, 3), delta=10)))
            videoB.append(dict(image=imageB, pts=curve.sample_spline((tckB[0], tckB[1].T, 3), delta=10)))
        video_pairs.append(videoA)
        video_pairs.append(videoB)

    return video_pairs


def split_fn(
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


class Guide3D(BaseGuide3D):
    """Guide3D dataset

    The dataset contains images and their corresponding t, c, u values,
    where:

    t: knot vector
    c: spline coefficients
    u: parameter values

    K, the degree of the spline, is 3.
    T, the knot vector, is of length n + k + 1, where n is the number of control
    points. The first k + 1 values are 0.
    """

    k = 3

    c_min = 0
    c_max = 1024
    t_min = 0
    t_max = 1274
    max_seq_len = 19

    def __init__(
        self,
        dataset_path: Union[str, Path],
        annotations_file: Union[str, Path] = "sphere_wo_reconstruct.json",
        image_transform: transforms.Compose = None,
        c_transform: callable = None,
        t_transform: callable = None,
        transform_both: callable = None,
        batch_first: bool = False,
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
        download: bool = False,
    ):
        super(Guide3D, self).__init__(
            dataset_path=dataset_path,
            annotations_file=annotations_file,
            process_data=process_data,
            split_fn=split_fn,
            download=download,
            split=split,
            split_ratio=split_ratio,
        )

        self.image_transform = image_transform
        self.c_transform = c_transform
        self.t_transform = t_transform
        self.transform_both = transform_both
        self.max_seq_len = self._get_max_length()

    def __len__(self):
        return len(self.data)

    def _get_max_length(self):
        max_length = 0
        for video in self.all_data:
            for sample in video:
                pts = sample["pts"]
                max_length = max(max_length, len(pts))
        return max_length

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = read_image(str(self.dataset_path / sample["image"]))

        pts = sample["pts"]

        if self.image_transform:
            img = self.image_transform(img)

        pts = torch.Tensor(pts).to(torch.float32)
        seq_len = pts.shape[0]
        target_seq = F.pad(pts, (0, 0, 0, self.max_seq_len - seq_len))

        target_mask = torch.ones(self.max_seq_len, dtype=torch.int32)
        target_mask[seq_len:] = 0

        return img, target_seq, target_mask


def quick_show(img, ts, cs, seq_len, index):
    import cv2
    import matplotlib.pyplot as plt
    from scipy.interpolate import splev

    img = img[index].squeeze(0).detach().cpu().numpy()
    seq_len = seq_len[index].detach().cpu().numpy().astype(int)
    ts = ts[index, :seq_len].detach().cpu().numpy()
    cs = cs[index, :seq_len].detach().cpu().numpy()

    img = cv2.resize(img, (256, 256))
    scale_factor = 256 / 1024

    ts = ts * scale_factor
    cs = cs * scale_factor

    ts = np.concatenate([np.zeros((4)), ts], axis=0)
    samples = np.linspace(0, ts[-1], 50)
    sampled_c = splev(samples, (ts, cs.T, 3))
    sampled_c = np.array(sampled_c).T

    plt.scatter(cs[:, 0], cs[:, 1], s=5)
    plt.plot(sampled_c[:, 0], sampled_c[:, 1], c="r")
    plt.imshow(img, cmap="gray")
    plt.show()


def test_dataset():
    import guide3d.vars as vars
    import matplotlib.pyplot as plt

    dataset_path = vars.dataset_path
    dataset = Guide3D(
        dataset_path,
        image_transform=image_transform,
    )
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    print(dataset._get_max_length())
    for batch in dataloader:
        img, target_seq, target_mask = batch
        plt.imshow(img[0][0], cmap="gray")
        print(target_seq.shape)
        plt.plot(target_seq[0][:, 0], target_seq[0][:, 1], "ro")
        plt.show()

        exit()
        continue
        print("Ts shape", ts.shape)
        print("Cs shape", cs.shape)
        print("T", ts.min(), ts.max())
        print("C", cs.min(), cs.max())
        continue
        print("Sequence_length", target_mask.sum(-1)[0])
        # print("C", cs.min(), cs.max())
        seq_len = target_mask.sum(dim=1)
        quick_show(img, ts, cs, seq_len, index=0)
        exit()
        # print(img.shape, target_seq.shape, target_mask.shape)


if __name__ == "__main__":
    test_dataset()
