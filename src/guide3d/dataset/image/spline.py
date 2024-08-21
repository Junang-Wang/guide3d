from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from guide3d.dataset.dataset_utils import BaseGuide3D
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

IMAGE_SIZE = 1024
N_CHANNELS = 1
MODEL_VERSION = "1"

vit_transform = transforms.Compose(
    [
        transforms.ToPILImage(),  # Convert image to PIL image
        # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize image to 224x224
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
        annotations_file: Union[str, Path] = "sphere.json",
        image_transform: transforms.Compose = None,
        c_transform: callable = None,
        t_transform: callable = None,
        add_init_token: bool = False,
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

        self.add_init_token = add_init_token
        self.max_length = self._get_max_length()

    def __len__(self):
        return len(self.data)

    def _get_ts(self):
        t_min = 0
        t_max = 0
        for video in self.all_data:
            for sample in video:
                t, c, _ = sample["tck"]
                t_min = min(t_min, t.min())
                t_max = max(t_max, t.max())

        return t_min, t_max

    def _get_cs(self):
        c_min = 0
        c_max = 0
        for video in self.all_data:
            for sample in video:
                t, c, _ = sample["tck"]
                c_min = min(c_min, c.min())
                c_max = max(c_max, c.max())

        return c_min, c_max

    def _get_max_length(self):
        max_length = 0
        for video in self.all_data:
            for sample in video:
                t, c, _ = sample["tck"]
                max_length = max(max_length, len(t) - 4)

        if self.add_init_token:
            max_length += 1
        return max_length

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = read_image(str(self.dataset_path / sample["image"]))

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

        target_seq = F.pad(torch.cat([t, c], dim=-1), (0, 0, 0, self.max_length - seq_len))

        target_mask = torch.ones(self.max_length, dtype=torch.int32)
        target_mask[seq_len:] = 0

        return img, target_seq, target_mask


def test_dataset():
    import guide3d.vars as vars

    dataset_path = vars.dataset_path
    dataset = Guide3D(
        dataset_path,
        "sphere_wo_reconstruct.json",
        image_transform=vit_transform,
    )
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    for batch in dataloader:
        img, target_seq, target_mask = batch
        ts = target_seq[:, :, 0]
        cs = target_seq[:, :, 1:]
        print("Ts shape", ts.shape)
        print("Cs shape", cs.shape)
        print("T", ts.min(), ts.max())
        print("Sequence_length", target_mask.sum(-1)[0])
        # print("C", cs.min(), cs.max())
        seq_len = target_mask.sum(dim=-1)
        # print(img.shape, target_seq.shape, target_mask.shape)


if __name__ == "__main__":
    test_dataset()
