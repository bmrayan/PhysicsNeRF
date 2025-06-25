import torch
import torch.nn.functional as F

def get_rays(H, W, focal, c2w):
    """Generate camera rays"""
    device = c2w.device
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='ij'
    )
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,3].expand(rays_d.shape)
    return rays_o, rays_d

def volume_render(raw, z_vals, rays_d, white_bkgd=False):
    """Basic volume rendering"""
    rgb = torch.sigmoid(raw[...,:3])
    alpha = F.relu(raw[...,3])
    
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=dists.device).expand(dists[...,:1].shape)], -1)
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
    
    alpha = 1.-torch.exp(-alpha * dists)
    T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.-alpha + 1e-10], -1), -1)[:,:-1]
    weights = alpha * T
    
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    
    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
    
    return rgb_map, depth_map, acc_map, weights
