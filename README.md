# PhysicsNeRF: Physics-Guided 3D Reconstruction from Sparse Views

**Anonymous Submission to ICML 2025 Workshop: Building Physically Plausible World Models**

## Abstract

We present PhysicsNeRF, a novel approach for physically consistent 3D reconstruction from sparse views by augmenting Neural Radiance Fields with physical priors. Our method integrates four complementary physics-based constraints: depth ranking supervision, RegNeRF consistency regularization, sparsity constraints, and cross-view consistency through a carefully designed 0.67M parameter architecture.

## Key Features

- **Sparse-view reconstruction** with only 8 training views
- **Physics-guided constraints** for geometric plausibility
- **Progressive training strategy** with curriculum learning
- **Balanced architecture design** (0.67M parameters)
- **Comprehensive evaluation** on static and dynamic scenes

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

## Installation

Clone repository
git clone https://github.com/anonymous-physics-nerf-2025/physics-sparse-nerf-anonymous.git
cd physics-sparse-nerf-anonymous

Install dependencies
pip install -r requirements.txt

Download NeRF synthetic data
wget https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
unzip nerf_synthetic.zip -d data/

## Quick Start

Train on Lego scene (8 views)
python train_physics_nerf.py --config configs/lego_config.json

Train on Chair scene
python train_physics_nerf.py --config configs/chair_config.json

Train on Drum scene
python train_physics_nerf.py --config configs/drum_config.json

Evaluate trained model
python evaluate.py --checkpoint checkpoints/best_model.pt --config configs/lego_config.json


## Training Details

- **Framework**: PyTorch 1.9+
- **Training Views**: 8 sparse views
- **Iterations**: 150k with progressive constraint scheduling
- **Hardware**: NVIDIA GPU with 8GB+ VRAM
- **Training Time**: ~6-8 hours

## Physics Constraint Scheduling

- **Phase 1 (0-5k)**: α=0.008 (gentle initialization)
- **Phase 2 (5k-15k)**: α=0.025 (light regularization)
- **Phase 3 (15k+)**: α=0.08 (full physics constraints)

## Key Findings

1. **Overfitting Challenge**: Generalization gaps of 4.7-7.2 dB demonstrate fundamental sparse-view limitations
2. **Training Dynamics**: Collapse-recovery patterns reveal optimization landscape complexities
3. **Physics Effectiveness**: Progressive constraints improve convergence but cannot eliminate overfitting
4. **Architecture Balance**: 0.67M parameters provide optimal capacity-generalization trade-off

## Citation
@article{anonymous2025physicsnerf,
title={PhysicsNeRF: Physics-Guided Neural Radiance Fields for Plausible 3D Reconstruction from Sparse Views},
author={Anonymous Authors},
journal={ICML 2025 Workshop on Building Physically Plausible World Models},
year={2025}
}


## Acknowledgments

This work builds upon NeRF, RegNeRF, and MiDaS. We thank the anonymous reviewers for their valuable feedback.

---

**Note**: This is a preliminary implementation for review purposes. Complete optimized implementation will be released upon paper acceptance.


