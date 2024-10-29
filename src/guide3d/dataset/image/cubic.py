from pathlib import Path
from typing import Dict, List, Union

import cv2
import guide3d.vars as vars
import matplotlib.pyplot as plt
import numpy as np
import torch
from guide3d.dataset.dataset_utils import BaseGuide3D
from guide3d.representations import curve
from torch.utils import data
from torchvision import transforms
from torchvision.io import read_image

IMAGE_SIZE = 1024
N_CHANNELS = 1
MODEL_VERSION = "1"

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


def preprocess_tck(
    tck: Dict,
) -> List:
    c = tck["c"]
    c = np.array(c)
    return c


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
    max_seq_len = 8

    def __init__(
        self,
        dataset_path: Union[str, Path],
        annotations_file: Union[str, Path] = "sphere_wo_reconstruct_bezier.json",
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
            split_fn=split_fn,
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
                c = sample["tck"]
                max_length = max(max_length, len(c))
        return max_length

    def __getitem__(self, idx):
        sample = self.data[idx]
        img = read_image(str(self.dataset_path / sample["image"]))

        c = sample["tck"]

        c = torch.tensor(c, dtype=torch.float32)

        if self.c_transform:
            c = self.c_transform(c)

        if self.image_transform:
            img = self.image_transform(img)

        seq_len = torch.tensor(len(c), dtype=torch.int32)
        target_seq = c

        target_mask = torch.ones(self.max_length, dtype=torch.int32)
        target_mask[seq_len:] = 0

        return img, target_seq, target_mask


def quick_show(img, ts, cs, seq_len, index):
    import matplotlib.pyplot as plt
    from scipy.interpolate import splev

    img = img[index].squeeze(0).detach().cpu().numpy()
    seq_len = seq_len[index].detach().cpu().numpy().astype(int)
    ts = ts[index, :seq_len].detach().cpu().numpy()
    cs = cs[index, :seq_len].detach().cpu().numpy()

    img = cv2.resize(img, (256, 256))
    scale_factor = 256 / 1024

    ts = ts * scale_factor
    cs = cs * scale_factor

    ts = np.concatenate([np.zeros((4)), ts], axis=0)
    samples = np.linspace(0, ts[-1], 50)
    sampled_c = splev(samples, (ts, cs.T, 3))
    sampled_c = np.array(sampled_c).T

    plt.scatter(cs[:, 0], cs[:, 1], s=5)
    plt.plot(sampled_c[:, 0], sampled_c[:, 1], c="r")
    plt.imshow(img, cmap="gray")
    plt.show()


def visualize_bezier(control_points, img):
    # Reconstruct the Bezier curve from the control points
    t_values = np.linspace(0, 1, 100)  # Parameter values for the smooth curve
    bezier_points = curve.bezier_curve(control_points, t_values)
    # Visualization
    plt.figure(figsize=(8, 6))

    # Plot the fitted Bézier curve
    plt.plot(bezier_points[:, 0], bezier_points[:, 1], "b-", label="Fitted Bézier Curve", linewidth=2)

    # Plot the control points and connect them with dashed lines
    plt.plot(control_points[:, 0], control_points[:, 1], "go--", label="Control Points", markersize=8)

    # Highlight control points with green circles
    for i, cp in enumerate(control_points):
        plt.text(cp[0], cp[1], f"P{i}", fontsize=12, color="green")

    # Label the axes and add a legend
    plt.title(f"Bézier Curve Fitting (Degree {len(control_points) - 1})")
    plt.imshow(img, cmap="gray")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.grid(True)

    # Display the plot
    plt.show()
    plt.close()


def test_dataset():
    dataset_path = vars.dataset_path
    dataset = Guide3D(
        dataset_path,
        image_transform=image_transform,
    )
    dataloader = data.DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(dataloader))
    print(dataset._get_max_length())
    for batch in dataloader:
        img, target_seq, target_mask = batch
        print(target_seq.shape)
        print(target_mask.shape)
        print(target_mask)
        visualize_bezier(target_seq[0].cpu().numpy(), img[0][0].cpu().numpy())
        continue
        print("Ts shape", ts.shape)
        print("Cs shape", cs.shape)
        print("T", ts.min(), ts.max())
        print("C", cs.min(), cs.max())
        continue
        print("Sequence_length", target_mask.sum(-1)[0])
        # print("C", cs.min(), cs.max())
        seq_len = target_mask.sum(dim=1)
        quick_show(img, ts, cs, seq_len, index=0)
        exit()
        # print(img.shape, target_seq.shape, target_mask.shape)


if __name__ == "__main__":
    test_dataset()
