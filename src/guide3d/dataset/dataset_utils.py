import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Union

import gdown
import numpy as np
from torch.utils import data


class Guide3D(data.Dataset):
    def __init__(
        self,
        dataset_path: Union[Path, str],
        annotations_file: Union[Path, str],
        process_data: callable,
        split_fn: callable,
        download: bool = False,
        split: str = "train",
        split_ratio: tuple = (0.8, 0.1, 0.1),
    ):
        assert split in [
            "train",
            "val",
            "test",
        ], "Split should be one of 'train', 'val', 'test'"

        assert len(split_ratio) == 3, f"Split ratio should be of length 3, got: {len(split_ratio)}"

        self.annotations_dir = Path(__file__).parent / "annotations"

        if isinstance(dataset_path, str):
            self.dataset_path = Path(dataset_path).expanduser().resolve()

        if not self.dataset_path.exists():
            if download:
                self._download_data()
            else:
                raise AssertionError(f"Path does not exist: {self.dataset_path}. Set download to True?")

        if isinstance(annotations_file, str):
            self.annotations_file = self.annotations_dir / annotations_file
            if not self.annotations_file.exists():
                raise AssertionError(
                    f"Annotation file should be one of: {[f for f in self.annotations_dir.iterdir()]}, got {self.annotations_file}"
                )
        else:
            self.annotations_file = annotations_file
            assert self.annotations_file.exists(), f"Annotations file not found: {self.annotations_file}"

        self.raw_data = json.load(open(self.annotations_file))
        data = process_data(self.raw_data)
        train_data, val_data, test_data = split_fn(data, split_ratio)

        if split == "train":
            self.data = train_data
        elif split == "val":
            self.data = val_data
        elif split == "test":
            self.data = test_data

    def _download_data(self):
        id = "11OcFDTwadJxhHKv9hJMxDL13eIqXMXSO"

        if not self.dataset_path.exists():
            self.dataset_path.mkdir()

        zip_path = self.dataset_path / "temp.zip"

        print("Downloading file with gdrive...")
        gdown.download(id=id, output=zip_path.as_posix(), quiet=False)

        print("Extracting contents...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.dataset_path)

        print(f"Extracted contents to {self.dataset_path}")

        os.remove(zip_path)


def flatten(
    annotations: List[List[Dict[str, Union[str, np.ndarray]]]],
) -> List[Dict[str, Union[str, np.ndarray]]]:
    flattened_annotations = []
    for video_pair in annotations:
        for frame in video_pair:
            flattened_annotations.append(
                dict(
                    img=frame["img1"]["path"],
                    pts=frame["img1"]["points"],
                    reconstruction=frame["reconstruction"],
                )
            )
            flattened_annotations.append(
                dict(
                    img=frame["img2"]["path"],
                    pts=frame["img2"]["points"],
                    reconstruction=frame["reconstruction"],
                )
            )
    return flattened_annotations


if __name__ == "__main__":
    dataset = Guide3D(
        dataset_path="~/test",
        annotations_file="sphere.json",
        split_fn=lambda x, y: (x, x, x),
        process_data=lambda x: x,
        download=True,
    )
