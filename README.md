# (REPO NOT FINISHED, COMEBACK IN A FEW DAYS) PhysicsNeRF: Physics-Guided Neural Radiance Fields for Plausible 3D Reconstruction from Sparse Views

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

## Results Summary

| Object | Train PSNR | Test PSNR | Gap (dB) |
|--------|-------------|------------|----------|
| Chair | 23.2 | 18.5 | 4.7 |
| Lego | 21.7 | 15.0 | 6.7 |
| Drums | 19.2 | 12.0 | 7.2 |
| **Average** | **21.4** | **15.2** | **6.2** |

## Method Overview

### Architecture
- Dual-scale coordinate processing (1× and 2×)
- D=7 layers, W=192 dimensions
- Moderate dropout (0.25) for regularization
- LayerNorm for training stability

### Physics Constraints
1. **Depth Ranking**: Monocular depth consistency using MiDaS
2. **RegNeRF Consistency**: Ray perturbation regularization  
3. **Sparsity**: Realistic density distributions
4. **Cross-View**: Multi-view geometric coherence

### Progressive Training
- Phase 1 (0-5k): α=0.008 (gentle start)
- Phase 2 (5k-15k): α=0.025 (light regularization)
- Phase 3 (15k+): α=0.08 (full constraints)


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


