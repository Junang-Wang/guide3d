---
title: Home
layout: home
nav_order: 1
---

# Guide3D

In endovascular surgery, accurate tool navigation is essential for improving patient outcomes and minimizing procedural risks. Key to this process is the reliable 3D reconstruction of surgical tools from fluoroscopic images. However, the development of machine learning models for this purpose has been limited by a lack of publicly available, high-quality datasets. Existing resources often use monoplanar fluoroscopic imaging due to the specialized and costly nature of bi-planar scanners, providing only a single perspective that restricts 3D reconstruction accuracy.

To address this gap, we introduce **Guide3D**—a bi-planar X-ray dataset tailored for advancing 3D reconstruction of endovascular surgical tools. Guide3D consists of high-resolution, manually annotated fluoroscopic videos, captured under real-world clinical conditions with bi-planar imaging systems. The dataset is validated within a simulated clinical environment, affirming its applicability for real-world applications and serving as a benchmark for guidewire shape prediction. This benchmark offers a robust baseline, setting the stage for future research in segmentation and 3D reconstruction of endovascular tools.

By providing Guide3D and accompanying code, we aim to empower the research community, facilitating the development of innovative machine learning techniques that enhance the precision and efficiency of endovascular procedures. Guide3D not only fills a crucial gap in available datasets but also paves the way for advancing safer, more effective interventions in endovascular surgery.

![Dataset Overview]({{ site.baseurl }}/assets/images/dataset_overview.jpg)

| Sample Type | Radifocus™ Guide Wire (Angled) | Nitrex Guidewire (Straight) | Total |
|-------------|--------------------------------|-----------------------------|-------|
| w fluid     | 3664                           | 484                         | 4148  |
| w/o fluid   | 2472                           | 2126                        | 4598  |
| **Total**   | **6136**                       | **2610**                    | **8746** |
