import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from guide3d.utils.utils import preprocess_tck
from torch.utils import data
from torchvision import transforms
from torchvision.io import read_image

image_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize((0.5,), (0.5,)),
        # gray to RGB
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    ]
)


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

            uA = np.array(frame["cameraA"]["u"]).astype(np.float32)
            uB = np.array(frame["cameraB"]["u"]).astype(np.float32)

            videoA.append(
                dict(
                    image=imageA,
                    tck=tckA,
                    u=uA,
                )
            )
            videoB.append(
                dict(
                    image=imageB,
                    tck=tckB,
                    u=uB,
                )
            )
        video_pairs.append(videoA)
        video_pairs.append(videoB)

    return video_pairs


def split_video_data(
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


class Guide3D(data.Dataset):
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

    def __init__(
        self,
        root: str,
        annotations_file: str = "sphere.json",
        image_transform: transforms.Compose = None,
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
        with_mask: bool = False,
        max_length: int = None,
        pad: bool = False,
    ):
        self.root = Path(root)
        self.annotations_file = annotations_file
        raw_data = json.load(open(self.root / self.annotations_file))
        data = process_data(raw_data)
        train_data, val_data, test_data = split_video_data(data, split_ratio)
        assert split in [
            "train",
            "val",
            "test",
        ], "Split should be one of 'train', 'val', 'test'"

        if split == "train":
            self.data = train_data
        elif split == "val":
            self.data = val_data
        elif split == "test":
            self.data = test_data

        self.image_transform = image_transform
        self.with_mask = with_mask
        if self.with_mask:
            assert (
                max_length is not None
            ), "max_length should be provided when with_mask is True"
        self.max_length = max_length
        self.pad = pad

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = read_image(str(self.root / sample["image"]))
        t, c, k = sample["tck"]

        seq_len = len(t) - 4  # first 4 values are 0
        seq_len = torch.tensor(seq_len, dtype=torch.int32)

        t = t[4:]  # first 4 values are 0
        t = np.array(t, dtype=np.float32)
        t = torch.tensor(t, dtype=torch.float32)
        t = F.pad(t, (0, self.max_length - len(t)))

        c = np.array(c, dtype=np.float32)
        c = torch.tensor(c, dtype=torch.float32)
        c = c.T
        c = F.pad(c, (0, 0, 0, self.max_length - len(c)))

        if self.image_transform:
            img = self.image_transform(img)

        if self.with_mask:
            mask = torch.ones(seq_len, dtype=torch.int32)
            mask[self.max_length :] = 0
            return img, t, c, seq_len, mask
        return img, t, c, seq_len, mask

    def decode_batch(self, batch, idx):
        img, t, c, seq_len, mask = batch
        img = img[idx].squeeze().numpy()
        seq_len = seq_len[idx].numpy()
        t = t[idx].numpy()[:seq_len]
        c = c[idx].numpy()[:seq_len]
        c = c.T

        mask = mask[idx].numpy()
        t = np.concatenate((np.zeros(4), t))

        return img, t, c, mask

    def visualize_sample(self, batch, idx, n_points=30):
        import cv2
        import matplotlib.pyplot as plt
        from guide3d.utils import viz
        from scipy.interpolate import splev

        img, t, c, mask = self.decode_batch(batch, idx)
        img = viz.convert_to_color(img)

        # draw control points
        for control_point in c.astype(np.int32).T:
            img = cv2.circle(img, tuple(control_point), 4, (255, 0, 0), -1)

        # draw spline
        sample_points = np.linspace(0, t[-1], n_points)
        spline_points = splev(sample_points, (t, c, self.k))
        for point in np.array(spline_points).astype(np.int32).T:
            img = cv2.circle(img, tuple(point), 2, (0, 255, 0), -1)

        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.show()


def test_dataset():
    import guide3d.vars as vars

    dataset_path = vars.dataset_path
    dataset = Guide3D(
        dataset_path, "sphere_wo_reconstruct.json", with_mask=True, max_length=50
    )
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=True)
    i = 0
    for batch in dataloader:
        img, t, c, mask = dataset.decode_batch(batch, 0)
        dataset.visualize_sample(batch, 0)

        if i % 10 == 0 and i != 0:
            exit()
        i += 1


if __name__ == "__main__":
    test_dataset()
