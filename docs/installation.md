---
layout: default
title: Installation
nav_order: 2
back_to_top: true
back_to_top_text: "Back to top"
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Installation

## Prerequisites

Guide3D requires minimal dependencies, relying only on PyTorch. It is
recommended, however, to use a `conda` environment.

```bash
conda create -n guide3d python==3.10
conda activate guide3d
```

## Guide3D

After the environment is created, the dataset can be simply installed using:

```bash
pip install git+https://github.com/airvlab/guide3d.git
```

At this point, you should be able to use the dataset. The dataset itself is
explained in [usage](https://github.com/airvlab/guide3d/usage).
