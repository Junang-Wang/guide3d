from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from torchvision.io import read_image

from guide3d.dataset.dataset_utils import BaseGuide3D
from guide3d.utils.utils import preprocess_tck, split_fn_image

IMAGE_SIZE = 1024
N_CHANNELS = 1

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

            videoA.append(dict(image=imageA, tck=tckA, u=uA))
            videoB.append(dict(image=imageB, tck=tckB, u=uB))
        video_pairs.append(videoA)
        video_pairs.append(videoB)

    return video_pairs


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

    def __init__(
        self,
        dataset_path: Union[str, Path],
        annotations_file: Union[str, Path] = "b_spline.json",
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
            split_fn=split_fn_image,
            download=download,
            split=split,
            split_ratio=split_ratio,
        )

        self.image_transform = image_transform
        self.c_transform = c_transform
        self.t_transform = t_transform
        self.transform_both = transform_both
        self.max_length = self._get_max_length()

    def __len__(self):
        return len(self.data)

    def _get_max_length(self):
        max_length = 0
        for video in self.all_data:
            for sample in video:
                t, c, _ = sample["tck"]
                max_length = max(max_length, len(t) - 4)
        return max_length

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = read_image(str(self.dataset_path / sample["image"]))

        t, c, _ = sample["tck"]

        # t has 4 zeros at the beginning
        t = torch.tensor(t[4:], dtype=torch.float32).unsqueeze(-1)
        c = torch.tensor(c, dtype=torch.float32)

        if self.transform_both:
            img, t, c = self.transform_both(img, t, c)

        if self.t_transform:
            t = self.t_transform(t)

        if self.c_transform:
            c = self.c_transform(c)

        if self.image_transform:
            img = self.image_transform(img)

        seq_len = torch.tensor(len(t), dtype=torch.int32)
        target_seq = F.pad(torch.cat([t, c], dim=-1), (0, 0, 0, self.max_length - seq_len))

        target_mask = torch.ones(self.max_length, dtype=torch.int32)
        target_mask[seq_len:] = 0

        return img, target_seq, target_mask


def main():
    import guide3d.vars as vars

    dataset = Guide3D(
        vars.dataset_path,
        image_transform=image_transform,
    )
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=False)
    print(len(dataset))
    batch = next(iter(dataloader))
    for batch in dataloader:
        img, target_seq, target_mask = batch
        exit()


if __name__ == "__main__":
    main()
