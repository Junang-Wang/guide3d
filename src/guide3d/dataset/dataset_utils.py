import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Union

import gdown
import numpy as np
from torch.utils import data


class BaseGuide3D(data.Dataset):
    """
    A custom PyTorch Dataset class for loading and processing Guide3D annotation data.

    This dataset class supports downloading the dataset if it does not exist locally,
    processing the data, and splitting it into train, validation, and test sets.

    Attributes:
        dataset_path (Path): Path to the dataset directory.
        annotations_file (Path): Path to the annotations file.
        raw_data (Dict): Raw data loaded from the annotations file.
        data (List): Processed data corresponding to the selected split.

    Args:
        dataset_path (Union[Path, str]): Path to the dataset directory. If the path is a string,
                                         it will be converted to a `Path` object.
        annotations_file (Union[Path, str]): Path to the JSON file containing annotations.
                                             If the path is a string, it will be resolved
                                             relative to the annotations directory.
        process_data (callable): Function to process the raw annotation data.
        split_fn (callable): Function to split the processed data into train, val, and test sets.
        download (bool, optional): If True, download the dataset if it does not exist locally.
                                   Defaults to False.
        split (str, optional): Which data split to load. Must be one of "train", "val", or "test".
                               Defaults to "train".
        split_ratio (tuple, optional): A tuple indicating the ratio of train, val, and test splits.
                                       Must have three elements. Defaults to (0.8, 0.1, 0.1).

    Raises:
        AssertionError: If the split is not one of "train", "val", or "test".
        AssertionError: If the length of split_ratio is not 3.
        AssertionError: If the dataset path does not exist and download is False.
        AssertionError: If the annotations file does not exist.
    """

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
        else:
            self.dataset_path = dataset_path

        if not self.dataset_path.exists():
            if download:
                self._download_data()
            else:
                raise AssertionError(f"Path does not exist: {self.dataset_path}. Set download to True?")

        if isinstance(annotations_file, str):
            self.annotations_file = self.annotations_dir / annotations_file
            if not self.annotations_file.exists():
                raise AssertionError(
                    f"Annotation file should be one of: {[f.name for f in self.annotations_dir.iterdir()]}, got {self.annotations_file.name}"
                )
        else:
            self.annotations_file = annotations_file
            assert self.annotations_file.exists(), f"Annotations file not found: {self.annotations_file}"

        self.raw_data = json.load(open(self.annotations_file))
        self.all_data = process_data(self.raw_data)
        train_data, val_data, test_data = split_fn(self.all_data, split_ratio)

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
    dataset = BaseGuide3D(
        dataset_path="~/test-2",
        annotations_file="sphere.json",
        split_fn=lambda x, y: (x, x, x),
        process_data=lambda x: x,
        download=True,
    )
