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
        c_transform: callable = None,
        t_transform: callable = None,
        add_init_token: bool = True,
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
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
        self.c_transform = c_transform
        self.t_transform = t_transform
        self.max_length = self._get_max_length()
        self.add_init_token = add_init_token

    def __len__(self):
        return len(self.data)

    def _get_max_length(self):
        max_length = 0
        for sample in self.data:
            t, c, _ = sample["tck"]
            max_length = max(max_length, len(t) - 4)
        return max_length

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = read_image(str(self.root / sample["image"]))

        t, c, _ = sample["tck"]

        # t has 4 zeros at the beginning
        t = torch.tensor(t[4:], dtype=torch.float32).unsqueeze(-1)
        c = torch.tensor(c, dtype=torch.float32)

        init_t = torch.tensor([0], dtype=torch.float32).unsqueeze(-1)
        init_c = torch.zeros_like(c[0]).unsqueeze(0)

        if self.add_init_token:
            t = torch.cat([init_t, t], dim=0)
            c = torch.cat([init_c, c], dim=0)

        if self.t_transform:
            t = self.t_transform(t)

        if self.c_transform:
            c = self.c_transform(c)

        seq_len = torch.tensor(len(t), dtype=torch.int32)

        if self.image_transform:
            img = self.image_transform(img)

        target_seq = F.pad(
            torch.cat([t, c], dim=-1), (0, 0, 0, self.max_length - seq_len)
        )

        target_mask = torch.ones(self.max_length, dtype=torch.int32)
        target_mask[seq_len:] = 0

        return img, target_seq, target_mask


def test_dataset():
    import guide3d.vars as vars

    dataset_path = vars.dataset_path
    dataset = Guide3D(dataset_path, "sphere_wo_reconstruct.json")
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True)
    batch = next(iter(dataloader))
    i = 0
    for batch in dataloader:
        img, target_seq, target_mask = batch
        seq_len = target_mask.sum(dim=-1)
        print(img.shape, target_seq.shape, target_mask.shape)
        # dataset.visualize_sample(batch, 0)

        # if i % 10 == 0 and i != 0:
        # exit()
        # i += 1


if __name__ == "__main__":
    test_dataset()
