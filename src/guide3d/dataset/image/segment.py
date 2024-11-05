from pathlib import Path
from typing import Dict, List, Union

import cv2
import numpy as np
from torch.utils import data
from torchvision import transforms

from guide3d.dataset.dataset_utils import BaseGuide3D
from guide3d.utils.utils import preprocess_tck, sample_spline, split_fn_image

image_transforms = transforms.Compose(
    [
        transforms.Lambda(lambda x: x / 255.0),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


def make_mask(tck, u, delta=0.1):
    """make a segmentation mask from a polyline"""
    tck = (tck[0], tck[1].T, tck[2])

    pts = sample_spline(tck, u, delta=delta).astype(np.int32)

    mask = np.zeros((1024, 1024), dtype=np.uint8)
    pts = np.array(pts, dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=2)
    mask = mask / 255
    mask = mask.astype(np.uint8)
    return mask


def process_data(data: Dict) -> List:
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
                    mask=make_mask(tckA, uA),
                )
            )
            videoB.append(
                dict(
                    image=imageB,
                    mask=make_mask(tckB, uB),
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
    def __init__(
        self,
        dataset_path: Union[str, Path],
        annotations_file: Union[str, Path] = "b_spline.json",
        image_transform: transforms.Compose = None,
        mask_transform: callable = None,
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
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = cv2.imread(str(self.dataset_path / sample["image"]), cv2.IMREAD_GRAYSCALE)
        mask = sample["mask"]
        if self.image_transform:
            img = self.image_transform(img)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        return img, mask


def main():
    import guide3d.vars as vars

    dataset = Guide3D(vars.dataset_path, "b_spline.json")
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    print(len(dataset))
    batch = next(iter(dataloader))
    for batch in dataloader:
        img, mask = batch
        print(img.shape)
        print(mask.shape)
        exit()


if __name__ == "__main__":
    main()
