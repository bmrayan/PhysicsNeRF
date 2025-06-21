# PhysicsNeRF: Physics-Guided Neural Radiance Fields for Plausible 3D Reconstruction from Sparse Views

**PhysicsNeRF** is a novel framework for physically consistent 3D reconstruction from sparse views, extending Neural Radiance Fields (NeRF) with physics-guided priors. It is designed for scenarios with only 8 fixed camera views and achieves state-of-the-art results on both static and dynamic scenes.

---

## Overview

Standard NeRFs struggle with sparse supervision, often producing implausible geometry and floating artifacts. **PhysicsNeRF** addresses this by integrating three complementary physical priors:
- **Depth ranking supervision:** Enforces local depth ordering using monocular depth estimates (MiDaS).
- **Frequency-aware regularization:** Prioritizes reconstruction of coarse, low-frequency structures before fine details using discrete wavelet transforms.
- **Semantic consistency:** Uses CLIP embeddings to ensure coherent object representation across viewpoints.

These priors restrict the solution space to physically plausible reconstructions, supporting generalizable world models for agent interaction, robotics, and simulation.

---

## Key Features

- **Sparse-view 3D reconstruction:** Achieves high-quality results using only 8 input views per scene.
- **Physics-based priors:** Enforces depth, frequency, and semantic constraints for plausible geometry.
- **Compact model:** 0.67M-parameter architecture with dual-scale coordinate encoding.
- **Progressive regularization:** Physics constraints are introduced in phases for stable optimization.
- **Supports static and dynamic scenes:** Extends to dynamic settings with temporal consistency.

---

## Results

### Static Scene Performance (8 input views)

| Method         | Lego (PSNR/SSIM) | Chair (PSNR/SSIM) | Drums (PSNR/SSIM) |
|:--------------:|:---------------:|:-----------------:|:-----------------:|
| NeRF           |  9.7 / 0.53     | 21.1 / 0.86       | 17.5 / 0.77       |
| RegNeRF        | 19.1 / 0.73     | 20.9 / 0.83       | 18.6 / 0.72       |
| DietNeRF       | 23.9 / 0.86     | 24.6 / 0.90       | 20.0 / 0.84       |
| **PhysicsNeRF**| **30.0 / 0.91** | **31.0 / 0.96**   | **28.4 / 0.93**   |

### Dynamic Scene Performance

| Dataset         | 10k iters | 50k iters | 200k iters |
|-----------------|-----------|-----------|------------|
| BouncingBalls   | 22.5/0.90 | 30.4/0.97 | 34.0/0.98  |
| JumpingJacks    | 29.8/0.96 | 33.5/0.98 | 37.0/0.98  |

> PhysicsNeRF outperforms prior methods, especially on challenging objects with thin structures and dynamic scenes, demonstrating the effectiveness of physics-based constraints[8].

---

## Method

### Physics-Guided Priors

- **Depth Ranking:** Uses MiDaS monocular depth to enforce ordinal relationships between pixel pairs, promoting physically consistent depth without requiring ground-truth.
- **Frequency-Aware Regularization:** Applies a discrete wavelet transform (DWT) to prioritize low-frequency (coarse) image structure before high-frequency details.
- **Semantic Consistency:** Enforces similarity in CLIP embedding space between rendered and ground-truth images, preserving object identity and semantics.

### Training Objective

The total loss combines RGB reconstruction with weighted physics priors:

\[
\mathcal{L} = \mathcal{L}_{\text{rgb}} + \lambda_d \mathcal{L}_{\text{depth}} + \lambda_w \mathcal{L}_{\text{dwt}} + \lambda_s \mathcal{L}_{\text{sem}}
\]

where recommended weights are \( \lambda_d = 0.05\text{–}0.2 \), \( \lambda_w = 0.1\text{–}0.5 \), \( \lambda_s = 0.01\text{–}0.1 \)[8].

### Progressive Regularization

Constraints are introduced in phases:
- **Phase 1 (0–5k iters):** Gentle initialization (\(\alpha=0.008\))
- **Phase 2 (5k–15k iters):** Light regularization (\(\alpha=0.025\))
- **Phase 3 (15k+ iters):** Full physics constraints (\(\alpha=0.08\))[1].

---

## Generalization

PhysicsNeRF achieves excellent training performance, but a known challenge is the gap between training and test views, especially for dynamic scenes. This reflects the fundamental difficulty of sparse-view 3D reconstruction. Ongoing work aims to further improve generalization through additional physics-based constraints[8].

---

## Installation

git clone https://github.com/bmrayan/PhysicsNeRF.git
cd PhysicsNeRF
pip install -r requirements.txt

---

## Usage

**Training:**
python train_physics_nerf.py --config configs/lego_config.json

**Evaluation:**
python evaluate.py --checkpoint checkpoints/best_model.pt --config configs/lego_config.json

---

## Citation

If you use this code, please cite:
@inproceedings{PhysicsNeRF2025,
title={PhysicsNeRF: Physics-Guided Neural Radiance Fields for Plausible 3D Reconstruction from Sparse Views},
author={Barhdadi, Mohamed Rayan and Kurban, Hasan and Alnuweiri, Hussein},
booktitle={ICML Workshop on Building Physically Plausible World Models},
year={2025},
note={arXiv:2505.23481}
}

---

## Impact

PhysicsNeRF advances physically plausible world modeling for robotics, AR, and simulation. By enforcing physical consistency, it helps mitigate dataset biases and supports reliable planning, manipulation, and temporal prediction in reconstructed environments.

---

For more details, see the [arXiv paper](https://arxiv.org/abs/2505.23481).

---


