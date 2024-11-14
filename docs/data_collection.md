---
layout: default
title: Data Collection
---

# Data Collection Setup

![Setup]({{ site.baseurl }}/assets/images/materials_overview.jpg)

## X-ray System

The Guide3D dataset was collected using a high-precision bi-planar X-ray system. This system incorporates:

- **Epsilon X-ray Generators**: 60kW output by EMD Technologies Ltd.
- **Image Intensifier Tubes**: 16-inch diameter by Thales, enhancing spatial resolution.
- **X-ray Tubes**: Varian dual focal spot, selected for increased resolution in both focal modes.
- **Collimators**: Ralco Automatic Collimators calibrated with acrylic mirrors and geometric alignment grids.

This bi-planar configuration enables imaging from two perpendicular planes, facilitating improved 3D reconstruction accuracy over traditional monoplanar systems.

## Anatomical Models

Our data collection utilizes a half-body vascular phantom from Elastrat Sarl Ltd., Switzerland, housed in a transparent, closed water circuit that mimics human blood flow dynamics. This model is constructed from soft silicone and incorporates continuous compact pumps to simulate real blood flow conditions. It is anatomically accurate, based on detailed postmortem vascular casts, reflecting human vasculature with high fidelity as documented in prominent research [^1][^2]. This anatomical model is central to achieving realistic simulations of vascular scenarios, enabling reliable endovascular tool tracking and navigation data.

<div class="d-flex flex-justify-around">
  <img src="{{ site.baseurl }}/assets/images/nitrex_guidewire.jpg" width="300"/>
  <img src="{{ site.baseurl }}/assets/images/radifocus_guidewire.jpg" width="300"/>
</div>

## Surgical Tools

Guide3D includes data collected with two commonly used guidewires in endovascular surgery, enhancing its relevance to real-world surgical procedures. The first guidewire, the Radifocus™ Guide Wire M Stiff Type from Terumo Ltd., is constructed from nitinol and coated with polyurethane-tungsten for visibility under X-ray imaging. With a 0.89 mm diameter, 260 cm length, and a 3 cm angled tip, it is optimized for navigating lesions and dissecting complex vasculature. The second guidewire, the Nitrex Guidewire by Nitrex Metal Inc., is also nitinol-based and features a gold-tungsten straight tip, measuring 0.89 mm in diameter and 400 cm in length, with a 15 cm tip. Its design facilitates stability during catheter exchange, serving as a supportive tool for more complex navigation tasks. These guidewires provide diverse operational data, enhancing the dataset’s versatility and applicability.

---

[^1]: Martin, J.-B., Sayegh, Y., Gailloud, P., Sugiu, K., Khan, H. G., Fasel, J. H., & Rüfenacht, D. A. (1998). _In-vitro models of human carotid atheromatous disease_.
[^2]: Gailloud, P., Pray, J. R., Muster, M., Piotin, M., Fasel, J. H. D., & Rüfenacht, D. A. (1997). An in vitro anatomic model of the human cerebral arteries with saccular arterial aneurysms. _Surg Radiol Anat_, Springer.
