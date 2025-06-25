import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """Standard positional encoding for NeRF"""
    
    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.freq_bands = 2.**torch.linspace(0., num_freqs-1, num_freqs)
        
    def forward(self, x):
        out = [x] if self.include_input else []
        for freq in self.freq_bands:
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, -1)
    
    def output_dim(self, input_dim):
        return input_dim * (1 + 2 * self.num_freqs) if self.include_input else input_dim * 2 * self.num_freqs
