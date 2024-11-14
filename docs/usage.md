---
layout: post
title: Usage
nav_order: 3
---


The dataset is organized with the following folder structure:

```bash
./guide3d
└── dataset
    ├── annotations
    │   ├── 3d.json
    │   ├── b_spline.json
    │   └── raw.json
    ├── dataset_utils.py
    ├── image
    │   ├── b_spline.py
    │   ├── points.py
    │   └── segment.py
    └── video
        └── b_spline.py
```

Within this structure:

- `annotations` contains JSON files with annotation data in various formats (`3d.json`, `b_spline.json`, and `raw.json`).
- `image` includes representations for the image dataset, organized into `segment`, `b_spline`, and `points` subtypes.
- `video` holds video-specific data and utility files.

Each dataset format is provided for easy import. To access the dataset, simply import the `Guide3D` class from the corresponding module.

Below is an example of loading and using the dataset with PyTorch's `DataLoader`:

```python
from torch.utils import data
from guide3d.dataset.image.segment import Guide3D

# Initialize the dataset
dataset = Guide3D(
    dataset_path="~/datasets/test",
    split="train",                # Specify the dataset split (train, val, test)
    split_ratio=(0.8, 0.1, 0.1),   # Define split ratios for train, val, and test
    download=True,                 # Set to True to download the dataset if not available locally
)

# Load the dataset with DataLoader
dataloader = data.DataLoader(dataset, batch_size=1)

# Iterate through the DataLoader
for batch in dataloader:
    img, mask = batch
    print("Image Shape:", img.shape)
    print("Mask Shape:", mask.shape)
    break
```
