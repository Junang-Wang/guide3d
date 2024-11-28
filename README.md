# Guide3D

## TODO's

- [x] Clean and document the segmentation code
- [x] Clean and document the image based code
- [x] Clean the repository
- [x] Add quick-start
- [ ] Add detailed explanation on README
- [ ] Explain the folder structure

## Quickstart

```python
from torch.utils import data

from guide3d.dataset.image.segment import Guide3D

dataset = Guide3D(
    dataset_path="~/datasets/test",
    annotations_file="raw.json",
    split="train",
    split_ratio=(0.8, 0.1, 0.1),
    download=True,
)

dataloader = data.DataLoader(dataset, batch_size=1)

batch = next(iter(dataloader))
for batch in dataloader:
    img, mask = batch
    print(img.shape)
    print(mask.shape)
    exit()
```


## Reference
If you are using our dataset, please cite us:

```bib
@article{jianu2024guide3d,
  title={Guide3D: A Bi-planar X-ray Dataset for 3D Shape Reconstruction},
  author={Jianu, Tudor and Huang, Baoru and Nguyen, Hoan and Bhattarai, Binod and Do, Tuong and Tjiputra, Erman and Tran, Quang and Berthet-Rayne, Pierre and Le, Ngan and Fichera, Sebastiano and others},
  journal={arXiv preprint arXiv:2410.22224},
  year={2024}
}
```