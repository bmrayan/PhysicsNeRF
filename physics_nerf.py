import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsNeRF(nn.Module):
    """Basic Physics-guided NeRF for sparse-view reconstruction"""
    
    def __init__(self, D=6, W=128, input_ch=60, input_ch_dir=24, skips=[3], 
                 use_dropout=True, dropout_rate=0.15):
        super().__init__()
        self.D = D
        self.W = W
        self.skips = skips
        self.input_ch = input_ch
        self.input_ch_dir = input_ch_dir
        
        # Main MLP layers
        self.pts_linears = nn.ModuleList()
        for i in range(D):
            if i == 0:
                self.pts_linears.append(nn.Linear(input_ch, W))
            elif i in skips:
                self.pts_linears.append(nn.Linear(input_ch + W, W))
            else:
                self.pts_linears.append(nn.Linear(W, W))
        
        # Output heads
        self.alpha_linear = nn.Linear(W, 1)
        self.feature_linear = nn.Linear(W, W)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_dir + W, W//2)])
        self.rgb_linear = nn.Linear(W//2, 3)
        
        # Regularization
        self.dropout = nn.Dropout(dropout_rate) if use_dropout else None
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([nn.LayerNorm(W) for _ in range(D)])
        
    def forward(self, x):
        input_pts, input_views = x[...,:self.input_ch], x[...,self.input_ch:]
        
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
            h = l(h)
            
            # Apply layer norm for training stability
            if h.shape[0] > 1:
                h = self.layer_norms[i](h)
            
            h = F.relu(h)
            
            # Apply dropout
            if self.dropout is not None and self.training and i > 1:
                h = self.dropout(h)
        
        # Outputs
        alpha = self.alpha_linear(h)
        feature = self.feature_linear(h)
        
        # RGB from view-dependent features
        h = torch.cat([feature, input_views], -1)
        for l in self.views_linears:
            h = F.relu(l(h))
        rgb = torch.sigmoid(self.rgb_linear(h))
        
        return torch.cat([rgb, alpha], -1)

class DualScaleNeRF(nn.Module):
    """Dual-scale processing for improved reconstruction"""
    
    def __init__(self, scales=[1, 2], base_ch=60, dir_ch=24, hidden_dim=128):
        super().__init__()
        self.scales = scales
        self.networks = nn.ModuleList([
            PhysicsNeRF(D=6, W=hidden_dim//max(s,1), input_ch=base_ch, 
                       input_ch_dir=dir_ch, skips=[2], use_dropout=True)
            for s in scales
        ])
        
        self.final_rgb = nn.Linear(3 * len(scales), 3)
        self.final_alpha = nn.Linear(len(scales), 1)
        
    def forward(self, x):
        features_rgb = []
        features_alpha = []
        
        for network, scale in zip(self.networks, self.scales):
            x_scaled = x.clone()
            x_scaled[..., :3] = x_scaled[..., :3] * scale
            
            feat = network(x_scaled)
            features_rgb.append(feat[..., :3])
            features_alpha.append(feat[..., 3:4])
        
        # Combine features
        rgb_concat = torch.cat(features_rgb, dim=-1)
        alpha_concat = torch.cat(features_alpha, dim=-1)
        
        rgb = torch.sigmoid(self.final_rgb(rgb_concat))
        alpha = self.final_alpha(alpha_concat)
        
        return torch.cat([rgb, alpha], -1)
