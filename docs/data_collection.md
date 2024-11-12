---
layout: default
title: Data Collection
---

# Data Collection Setup

![Setup]({{ site.baseurl }}/assets/images/materials_overview.jpg)

## X-ray System

The Guide3D dataset was collected using a high-precision bi-planar X-ray system. This system incorporates:

- **Epsilon X-ray Generators**: $60$kW output by EMD Technologies Ltd.
- **Image Intensifier Tubes**: $16$-inch diameter by Thales, enhancing spatial resolution.
- **X-ray Tubes**: Varian dual focal spot, selected for increased resolution in both focal modes.
- **Collimators**: Ralco Automatic Collimators calibrated with acrylic mirrors and geometric alignment grids.

This bi-planar configuration enables imaging from two perpendicular planes, facilitating improved 3D reconstruction accuracy over traditional monoplanar systems.

## Anatomical Models

Our data collection utilizes a half-body vascular phantom from Elastrat Sarl Ltd., Switzerland, housed in a transparent, closed water circuit that mimics human blood flow dynamics. This model is constructed from soft silicone and incorporates continuous compact pumps to simulate real blood flow conditions. It is anatomically accurate, based on detailed postmortem vascular casts, reflecting human vasculature with high fidelity as documented in prominent research~\cite{martin1998vitro,gailloud1999vitro}. This anatomical model is central to achieving realistic simulations of vascular scenarios, enabling reliable endovascular tool tracking and navigation data.

<div class="d-flex flex-justify-around">
  <img src="{{ site.baseurl }}/assets/images/nitrex_guidewire.jpg" width="300"/>
  <img src="{{ site. baseurl }}/assets/images/radifocus_guidewire.jpg" width="300"/>
</div>

## Surgical Tools

Guide3D includes data collected with two commonly used guidewires in endovascular surgery, enhancing its relevance to real-world surgical procedures. The first guidewire, the Radifocus™ Guide Wire M Stiff Type from Terumo Ltd. (Fig.~\ref{fig:guidewire-radifocus}), is constructed from nitinol and coated with polyurethane-tungsten for visibility under X-ray imaging. With a \(0.89 \unit{\milli\meter}\) diameter, \(260 \unit{\centi\meter}\) length, and a \(3 \unit{\centi\meter}\) angled tip, it is optimized for navigating lesions and dissecting complex vasculature. The second guidewire, the Nitrex Guidewire by Nitrex Metal Inc. (Fig.~\ref{fig:guidewire-nitrex}), is also nitinol-based and features a gold-tungsten straight tip, measuring \(0.89 \unit{\milli\meter}\) in diameter and \(400 \unit{\centi\meter}\) in length, with a \(15 \unit{\centi\meter}\) tip. Its design facilitates stability during catheter exchange, serving as a supportive tool for more complex navigation tasks. These guidewires provide diverse operational data, enhancing the dataset’s versatility and applicability.

# Data Acquisition, Labeling, and Statistics

Using the materials specified, we constructed a dataset comprising 8,746 high-resolution images (with a resolution of \(1,024 \times 1,024\) pixels). This collection includes 4,373 paired instances with and without simulated blood flow to capture diverse procedural conditions. Specifically, the dataset contains 6,136 samples from the Radifocus guidewire and 2,610 from the Nitrex guidewire, establishing a robust basis for automated guidewire tracking within bi-planar images. All annotations were performed manually using the Computer Vision Annotation Tool (CVAT)~\cite{cvat2023}, where the guidewire paths were traced with polylines to capture their curvilinear nature. This choice enables accurate tracking for looping or overlapping sections that would be difficult to represent with segmentation masks.

As summarized in Table~\ref{tab:dataset-stats}, the dataset includes 3,664 angled guidewire instances with simulated blood flow and 484 without, while straight guidewires are represented by 2,472 instances with fluid and 2,126 without. This balanced distribution across guidewire types and procedural contexts supports a wide range of training scenarios. Each of the 8,746 images includes manually generated segmentation ground truth data, providing a foundational resource for developing and validating segmentation and tracking algorithms.

# Guidewire Reconstruction

Given polyline representations of a guidewire curve in both X-ray planes, the reconstruction process in Guide3D involves parameterizing these curves through B-Spline interpolation. We express each curve as a function of cumulative distance along the curve, with each plane represented by parameterized B-Spline curves \( \mathbf{C}_A(\mathbf{u}_A) \) and \( \mathbf{C}_B(\mathbf{u}_B) \), where \( \mathbf{u}_A \) and \( \mathbf{u}_B \) denote normalized arc-length parameters. Using epipolar geometry, we find the corresponding \( \mathbf{u}_B \) for each \( \mathbf{u}_A \). Once matched, the 3D coordinates \( \mathbf{P}^i \) of these points are computed via triangulation, resulting in a set of 3D points \( \{\mathbf{P}^i\}_{i=1}^{M} \) that accurately reconstruct the guidewire in 3D space.

## Retrieving the Fundamental Matrix \( \mathbf{F} \)

The relationship between corresponding points in Images A (\( \mathbf{I}_A \)) and B (\( \mathbf{I}_B \)) is defined by the fundamental matrix \( \mathbf{F} \), which satisfies \( \mathbf{x}_B^T \mathbf{F} \mathbf{x}_A = 0 \). Following the calibration detailed in Subsection \ref{subsec:calibration}, we have the projection matrices \( \mathbf{P}_A \) and \( \mathbf{P}_B \). The fundamental matrix is derived as:

\[
\mathbf{F} = [\mathbf{e}_B]_\times \mathbf{P}_B \mathbf{P}_A^+  
\]

where \( \mathbf{e}_B \) is the epipole in Image B, given by \( \mathbf{e}_B = \mathbf{P}_B \mathbf{C}_A \), and \( [\mathbf{e}_B]_\times \) denotes the skew-symmetric matrix of \( \mathbf{e}_B \):

\[
[\mathbf{e}_B]_\times = \begin{bmatrix}
0 & -e_{B3} & e_{B2} \\
e_{B3} & 0 & -e_{B1} \\
-e_{B2} & e_{B1} & 0
\end{bmatrix}
\]

Here, \( \mathbf{e}_B = (e_{B1}, e_{B2}, e_{B3})^T \), and \( \mathbf{P}_A^+ \) represents the pseudoinverse of \( \mathbf{P}_A \). The matrix \( \mathbf{F} \) encapsulates the epipolar geometry, ensuring that corresponding points \( \mathbf{x}_A \) and \( \mathbf{x}_B \) lie on each other's epipolar lines.

## Matching

The matching phase begins by uniformly sampling points along \( \mathbf{C}_A(u_A) \) at intervals \( \Delta u_A \). For each sampled point \(x_A = \mathbf{C}_A(u_A)\), we project the epiline \( l_B = F x_A \) onto Image B, finding intersections between \( l_B \) and \( \mathbf{C}_B(u_B) \) to determine the corresponding parameter \( u_B \).

In cases where \( l_B \) does not intersect with \( \mathbf{C}_B \) due to projection matrix inaccuracies, we employ Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) to fit a monotonic function \( f_A(u_A) \rightarrow u_B \), as in~\cite{altingovde20223d}. This approach interpolates missing intersections, with the matching process illustrated in Fig.~\ref{fig:matching}.

<!-- ![Point Matching Process: Sampled points from image \(I_A\) (\(C_A(u_A)\)) and their corresponding epilines \(l_A\) on image \(I_B\) with their matches \(C_B(u_B)\). The epilines for \(C_B(u_B)\) are then computed and displayed on the image \(I_A\).](assets/matching.jpg) -->
_Figure: Point Matching Process_

## Triangulation

With projections from both cameras, we form a system of linear equations:

\[
\begin{aligned}
    \mathbf{x}_1 \times (\mathbf{P}_1 \mathbf{X}) = 0  \\
    \mathbf{x}_2 \times (\mathbf{P}_2 \mathbf{X}) = 0
\end{aligned}
\]

This creates a \( 4 \times 4 \) matrix \( \mathbf{A} \) derived from projection matrices \( \mathbf{P}_1 \) and \( \mathbf{P}_2 \) and image coordinates \( \mathbf{x}_A \) and \( \mathbf{x}_B \). Applying Singular Value Decomposition (SVD)~\cite{klema1980singular} to \( \mathbf{A} \), we determine \( \mathbf{X} \), the eigenvector associated with \( \mathbf{A} \)'s smallest singular value, representing the triangulated 3D point in homogeneous coordinates.

To ensure accurate reconstruction, all reprojections undergo manual verification to address potential issues like partial occlusion or suboptimal acquisition angles, which may cause incorrect or missing points. Samples with inaccurate reconstruction are manually discarded to maintain dataset quality.
