from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.io import read_image

from guide3d.dataset.dataset_utils import BaseGuide3D
from guide3d.utils.utils import preprocess_tck, sample_spline

IMAGE_SIZE = 1024
N_CHANNELS = 1

image_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(N_CHANNELS, 1, 1)),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]
)


def filter_pts(pts):
    return np.array([pt for pt in pts if 0 <= pt[0] < IMAGE_SIZE and 0 <= pt[1] < IMAGE_SIZE])


def process_data(data: Dict) -> List:
    video_pairs = []
    for video_pair in data:
        videoA, videoB = [], []
        for frame in video_pair["frames"]:
            imageA = frame["cameraA"]["image"]
            imageB = frame["cameraB"]["image"]

            tckA = preprocess_tck(frame["cameraA"]["tck"])
            tckB = preprocess_tck(frame["cameraB"]["tck"])

            videoA.append({"image": imageA, "pts": sample_spline((tckA[0], tckA[1].T, 3), delta=10)})
            videoB.append({"image": imageB, "pts": sample_spline((tckB[0], tckB[1].T, 3), delta=10)})
        video_pairs.extend([videoA, videoB])

    return video_pairs


def split_fn(data: List, split: tuple = (0.8, 0.1, 0.1)) -> List:
    train_data, val_data, test_data = [], [], []
    for video in data:
        train_idx = int(split[0] * len(video))
        val_idx = int(split[1] * len(video))
        train_data.extend(video[:train_idx])
        val_data.extend(video[train_idx : train_idx + val_idx])
        test_data.extend(video[train_idx + val_idx :])
    return train_data, val_data, test_data


class Guide3D(BaseGuide3D):
    """Guide3D dataset with images and spline points"""

    k = 3
    max_seq_len = 19

    def __init__(
        self,
        dataset_path: Union[str, Path],
        annotations_file: Union[str, Path] = "b_spline.json",
        image_transform: transforms.Compose = None,
        c_transform: callable = None,
        t_transform: callable = None,
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
        self.max_seq_len = self._get_max_length()

    def __len__(self):
        return len(self.data)

    def _get_max_length(self):
        return max(len(sample["pts"]) for video in self.all_data for sample in video)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = read_image(str(self.dataset_path / sample["image"]))
        pts = torch.Tensor(sample["pts"]).to(torch.float32)
        if self.image_transform:
            img = self.image_transform(img)

        seq_len = pts.shape[0]
        target_seq = F.pad(pts, (0, 0, 0, self.max_seq_len - seq_len))
        target_mask = torch.ones(self.max_seq_len, dtype=torch.int32)
        target_mask[seq_len:] = 0

        return img, target_seq, target_mask


def main():
    from torch.utils import data

    import guide3d.vars as vars

    dataset = Guide3D(
        dataset_path=vars.dataset_path,
        image_transform=image_transform,
    )
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)


if __name__ == "__main__":
    main()
