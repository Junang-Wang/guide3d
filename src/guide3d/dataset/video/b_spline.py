from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import io

from guide3d.dataset.dataset_utils import BaseGuide3D
from guide3d.utils import utils


def process_data(
    data: Dict,
    seq_len: int = 3,
    cameras: str = "A",
) -> List:
    videos = []
    for video_pair in data:
        videoA = []
        videoB = []
        for frame in video_pair["frames"]:
            imageA = frame["cameraA"]["image"]
            imageB = frame["cameraB"]["image"]

            tckA = utils.preprocess_tck(frame["cameraA"]["tck"])
            tckB = utils.preprocess_tck(frame["cameraB"]["tck"])

            uA = np.array(frame["cameraA"]["u"])
            uB = np.array(frame["cameraB"]["u"])

            tck3d = utils.preprocess_tck(frame["3d"]["tck"])
            u3d = np.array(frame["3d"]["u"])

            videoA.append(
                dict(
                    image=imageA,
                    tck=tckA,
                    u=uA,
                    tck3d=tck3d,
                    u3d=u3d,
                )
            )
            videoB.append(
                dict(
                    image=imageB,
                    tck=tckB,
                    u=uB,
                    tck3d=tck3d,
                    u3d=u3d,
                )
            )

        if "A" in cameras:
            videos.append(videoA)
        if "B" in cameras:
            videos.append(videoB)

    new_videos = []
    for video in videos:
        new_video = []
        for i in range(0, len(video) - seq_len + 1):
            new_video.append(video[i : i + seq_len])
        new_videos.append(new_video)

    return new_videos


class Guide3D(BaseGuide3D):
    def __init__(
        self,
        dataset_path: Union[str, Path],
        annotations_file: Union[str, Path] = "3d.json",
        image_transform: transforms.Compose = None,
        pts_transform: callable = None,
        seg_len: int = 3,
        max_len: int = 150,
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
        download: bool = False,
    ):
        super(Guide3D, self).__init__(
            dataset_path=dataset_path,
            annotations_file=annotations_file,
            process_data=process_data,
            split_fn=utils.split_fn_video,
            download=download,
            split=split,
            split_ratio=split_ratio,
        )

        self.image_transform = image_transform
        self.pts_transform = pts_transform

        self.seg_len = seg_len
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_entry = self.data[index]
        imgs_paths = [self.root / img_path for img_path in data_entry["image"]]
        imgs = [io.read_image(img_path.as_posix()) for img_path in imgs_paths]

        if self.image_transform:
            imgs = [self.image_transform(img) for img in imgs]

        imgs = torch.stack(imgs, dim=0)

        pts = data_entry["pts_reconstructed"]
        if self.pts_transform:
            pts = [self.pts_transform(pt) for pt in pts]

        lengths = [torch.tensor(len(points), dtype=torch.int32) for points in pts]
        lengths = torch.stack(lengths, dim=0)

        pts = [torch.tensor(pt, dtype=torch.float32) for pt in pts]
        pts = [torch.cat((pt, torch.zeros((self.max_len - len(pt), 3)))) for pt in pts]
        pts = torch.stack(pts, dim=0)

        return imgs, pts, lengths


def main():
    import guide3d.vars as vars

    dataset = Guide3D(vars.dataset_path, split="train")
    print(len(dataset))
    sample = dataset[0]


if __name__ == "__main__":
    main()
