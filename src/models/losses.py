import torch
import torch.nn.functional as F
import numpy as np

def depth_ranking_loss(pred_depth, target_depth, n_samples=400):
    """Basic depth ranking consistency loss"""
    if isinstance(pred_depth, torch.Tensor):
        pred_depth = pred_depth.detach().cpu().numpy()
    
    pred_flat = pred_depth.flatten()
    target_flat = target_depth.flatten()
    
    n = len(pred_flat)
    if n < n_samples:
        return 0.0
    
    idx = np.random.choice(n, size=n_samples, replace=False)
    i_idx = idx[:n_samples//2]
    j_idx = idx[n_samples//2:]
    
    pred_diff = pred_flat[i_idx] - pred_flat[j_idx]
    target_diff = target_flat[i_idx] - target_flat[j_idx]
    
    inconsistent = ((pred_diff > 0) != (target_diff > 0)).astype(np.float32)
    
    return float(np.mean(inconsistent))

def sparsity_loss(density, lambda_sparse=0.01):
    """Sparsity regularization for realistic density"""
    return lambda_sparse * torch.mean(F.softplus(density))

def consistency_loss(rgb1, rgb2, lambda_consist=0.03):
    """Basic consistency loss between views"""
    return lambda_consist * F.mse_loss(rgb1, rgb2)
