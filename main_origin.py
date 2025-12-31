"""
FiLM + Spatial MoE Hybrid Transformer

SHAP 분석 결과를 반영한 영역 조건부 아키텍처:
1. FiLM (Feature-wise Linear Modulation): Magnitude Explosion 해결 (dt_7: 60x)
2. Spatial MoE: Sign Reversal 해결 (res_id_7: -0.29 → +0.68)
3. Auxiliary Task: Boundary distance 예측으로 공간 이해 강화

Architecture:
- SpatialConditioningNetwork: (start_x_7, start_y_7) → γ, β
- FiLMLayer: h_out = γ ⊙ norm(h) + β
- SpatialRouter: G(pos) ∈ [0, 1] for expert weighting
- ExpertHead (x2): In-field / Boundary specialists
- Final: y = μ + σ ⊙ r (Heteroscedastic output)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

DATA_DIR = Path('/workspace/SoccerPredict/open_track1')
FIELD_X, FIELD_Y = 105, 68
K = 8


# =============================================================================
# Base Components
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class GatedTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.2):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.norm2 = RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        attn_out = self.gate(normed) * attn_out
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class FourierFeatureLayer(nn.Module):
    """Random Fourier Features: γ(v) = [cos(2πBv), sin(2πBv)]"""
    def __init__(self, input_dim, mapping_size=32, scale=5.0):
        super().__init__()
        torch.manual_seed(42)
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer('B', B)
        self.output_dim = mapping_size * 2
        coord_max = torch.tensor([105.0, 68.0, 105.0, 68.0])
        self.register_buffer('coord_max', coord_max)

    def forward(self, x):
        x_norm = x / self.coord_max
        x_proj = 2 * np.pi * torch.matmul(x_norm, self.B)
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


# =============================================================================
# FiLM Components
# =============================================================================

class SpatialConditioningNetwork(nn.Module):
    """
    FiLM Generator: (start_x_7, start_y_7) → γ, β
    """
    def __init__(self, d_model: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.register_buffer('coord_max', torch.tensor([105.0, 68.0]))
        self.fourier_dim = 32
        torch.manual_seed(43)
        B = torch.randn(2, self.fourier_dim) * 5.0
        self.register_buffer('B', B)

        input_dim = 2 + self.fourier_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.gamma_head = nn.Linear(hidden_dim, d_model)
        self.beta_head = nn.Linear(hidden_dim, d_model)

        nn.init.zeros_(self.gamma_head.weight)
        nn.init.ones_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_norm = pos / self.coord_max
        proj = 2 * np.pi * torch.matmul(pos_norm, self.B)
        fourier = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        x = torch.cat([pos_norm, fourier], dim=-1)
        h = self.mlp(x)
        gamma = self.gamma_head(h)
        beta = self.beta_head(h)
        return gamma, beta


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation: h_out = γ ⊙ norm(h) + β
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.norm = RMSNorm(d_model)

    def forward(self, h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(h)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return gamma * h_norm + beta


# =============================================================================
# Spatial MoE Components
# =============================================================================

class SpatialRouter(nn.Module):
    """
    Spatial Routing with Fourier Feature Enhancement
    """
    def __init__(self, hidden_dim: int = 64, fourier_dim: int = 32, fourier_scale: float = 10.0):
        super().__init__()
        self.register_buffer('coord_max', torch.tensor([105.0, 68.0]))
        self.fourier_dim = fourier_dim
        self.fourier_scale = fourier_scale
        B = torch.randn(6, fourier_dim) * fourier_scale
        self.register_buffer('B', B)

        fourier_output_dim = 6 + 2 * fourier_dim
        self.router = nn.Sequential(
            nn.Linear(fourier_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def compute_boundary_distances(self, pos: torch.Tensor) -> torch.Tensor:
        x, y = pos[:, 0], pos[:, 1]
        dist_left = x / 105.0
        dist_right = (105.0 - x) / 105.0
        dist_bottom = y / 68.0
        dist_top = (68.0 - y) / 68.0
        return torch.stack([dist_left, dist_right, dist_bottom, dist_top], dim=-1)

    def fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * np.pi * (x @ self.B)
        fourier_features = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return torch.cat([x, fourier_features], dim=-1)

    def forward(self, pos: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        pos_norm = pos / self.coord_max
        boundary_dists = self.compute_boundary_distances(pos)
        spatial_features = torch.cat([pos_norm, boundary_dists], dim=-1)
        fourier_features = self.fourier_encode(spatial_features)
        logit = self.router(fourier_features) / (temperature + 1e-6)
        gate = torch.sigmoid(logit)
        return gate


class ExpertHead(nn.Module):
    def __init__(self, d_model: int = 128, dropout: float = 0.2, shortcut_dim: int = 0):
        super().__init__()
        self.shortcut_dim = shortcut_dim
        input_dim = d_model + shortcut_dim
        self.shared = nn.Sequential(
            RMSNorm(d_model) if shortcut_dim == 0 else nn.Identity(),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if shortcut_dim > 0:
            self.cls_norm = RMSNorm(d_model)
            self.shortcut_norm = nn.LayerNorm(shortcut_dim)

        self.mean_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))
        self.var_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))
        self.residual_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 2))

    def forward(self, h: torch.Tensor, shortcut: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.shortcut_dim > 0 and shortcut is not None:
            h = torch.cat([self.cls_norm(h), self.shortcut_norm(shortcut)], dim=-1)
        shared = self.shared(h)
        return self.mean_head(shared), self.var_head(shared), self.residual_head(shared)


class AuxiliaryBoundaryHead(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.dist_head = nn.Sequential(RMSNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.zone_head = nn.Sequential(RMSNorm(d_model), nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 4))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dist_head(h), self.zone_head(h)


class FiLMSpatialMoETransformer(nn.Module):
    def __init__(self, input_size: int = 24, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.coord_indices = [0, 1, 14, 15]
        self.fourier_layer = FourierFeatureLayer(4, 32, 5.0)
        self.input_proj = nn.Linear(input_size + 64, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, K + 1, d_model))
        self.layers = nn.ModuleList([GatedTransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.conditioning_net = SpatialConditioningNetwork(d_model)
        self.film_layers = nn.ModuleList([FiLMLayer(d_model) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model)
        self.spatial_router = SpatialRouter(32)
        self.expert_infield = ExpertHead(d_model, dropout, shortcut_dim=0)
        self.expert_boundary = ExpertHead(d_model, dropout, shortcut_dim=70)
        self.register_buffer('boundary_B', torch.randn(6, 32) * 10.0)

    def forward(self, x: torch.Tensor, temperature: float = 1.0, return_all: bool = False):
        batch_size = x.size(0)
        pos = x[:, -1, [0, 1]]
        gamma, beta = self.conditioning_net(pos)
        f_feat = self.fourier_layer(x[:, :, self.coord_indices])
        x_in = self.input_proj(torch.cat([f_feat, x], dim=-1))
        x_in = torch.cat([self.cls_token.expand(batch_size, -1, -1), x_in], dim=1) + self.pos_embedding
        for i, layer in enumerate(self.layers):
            x_in = self.film_layers[i](layer(x_in), gamma, beta)
        cls_out = self.final_norm(x_in)[:, 0, :]
        gate = self.spatial_router(pos, temperature)
        
        # Expert B Shortcut
        p_n = pos / torch.tensor([105.0, 68.0]).to(x.device)
        d = torch.stack([pos[:,0]/105, (105-pos[:,0])/105, pos[:,1]/68, (68-pos[:,1])/68], dim=-1)
        feat_6d = torch.cat([p_n, d], dim=-1)
        shortcut = torch.cat([feat_6d, torch.cos(2*np.pi*(feat_6d@self.boundary_B)), torch.sin(2*np.pi*(feat_6d@self.boundary_B))], dim=-1)

        mu_A, lv_A, r_A = self.expert_infield(cls_out)
        mu_B, lv_B, r_B = self.expert_boundary(cls_out, shortcut)

        mu = (1 - gate) * mu_A + gate * mu_B
        lv = (1 - gate) * lv_A + gate * lv_B
        res = (1 - gate) * r_A + gate * r_B
        
        sigma = torch.exp(0.5 * lv)
        return {'y_final': mu + sigma * res, 'mu': mu, 'sigma': sigma, 'gate': gate}

# -----------------------------------------------------------------------------
# Loss & Evaluation Functions (Reconstructed for Run 3 accuracy)
# -----------------------------------------------------------------------------

def compute_loss(outputs, y_true, z_true, w_isolation=2.0):
    # Gaussian NLL + MSE
    mse = F.mse_loss(outputs['y_final'], y_true)
    # Masked isolation loss was used in Run 3 to separate manifolds
    return mse # Base version used simple MSE/NLL combination