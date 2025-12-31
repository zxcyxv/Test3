"""
FiLM + Spatial MoE Hybrid Transformer

SHAP 분석 결과를 반영한 영역 조건부 아키텍처:
1. FiLM (Feature-wise Linear Modulation): Magnitude Explosion 해결 (dt_7: 60x)
2. Spatial MoE: Sign Reversal 해결 (res_id_7: -0.29 → +0.68)
3. Auxiliary Task: Boundary distance 예측으로 공간 이해 강화

Architecture:
- SpatialConditioningNetwork: (start_x_7, start_y_7) → γ, β
- FiLMLayer: h_out = γ ⊙ norm(h) + β
- SpatialRouter: G(pos) ∈ [0,1] for expert weighting
- ExpertHead (x2): In-field / Boundary specialists
- Final: y = μ + σ ⊙ r (Heteroscedastic output)
"""

import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
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
MAX_LEN = 105.0

# Feature groups
COORD_FEATURES = ['start_x', 'start_y', 'end_x', 'end_y']
ANGLE_FEATURES = ['angle_to_goal', 'action_angle', 'angle_visible']
ANGLE_FLIP_FEATURES = ['angle_to_goal', 'action_angle']
CAT_FEATURES = ['type_id', 'res_id', 'is_home', 'x_zone', 'lane', 'is_zone14']
CONT_FEATURES = [
    'dt', 'ep_idx_norm', 'dist_to_goal', 'pressure_x_weight',
    'dx', 'dy', 'dist', 'speed',
    'action_progress', 'action_dist', 'action_lateral',
]


# =============================================================================
# Base Components (from heteroscedastic model)
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

    def forward(self, x, src_key_padding_mask: Optional[torch.Tensor] = None):
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed, key_padding_mask=src_key_padding_mask
        )
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

    def forward(self, x):
        x_proj = 2 * np.pi * torch.matmul(x, self.B)
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


# =============================================================================
# FiLM Components
# =============================================================================

class SpatialConditioningNetwork(nn.Module):
    """
    FiLM Generator: (start_x_7, start_y_7) → γ, β

    위치 정보를 기반으로 Transformer의 특성 추출을 제어하는
    scale(γ)과 shift(β) 파라미터를 실시간 생성
    """
    def __init__(self, d_model: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.d_model = d_model

        # Fourier features for position
        self.fourier_dim = 32
        torch.manual_seed(43)  # Different seed from main Fourier layer
        B = torch.randn(2, self.fourier_dim) * 5.0
        self.register_buffer('B', B)

        # Input: 2 raw + 64 Fourier = 66
        input_dim = 2 + self.fourier_dim * 2

        # Conditioning MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Separate heads for gamma and beta
        self.gamma_head = nn.Linear(hidden_dim, d_model)
        self.beta_head = nn.Linear(hidden_dim, d_model)

        # Initialize: gamma=1 (identity), beta=0 (no shift)
        nn.init.zeros_(self.gamma_head.weight)
        nn.init.ones_(self.gamma_head.bias)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pos: [batch, 2] - (start_x_7, start_y_7)
        Returns:
            gamma: [batch, d_model] - scale parameters
            beta: [batch, d_model] - shift parameters
        """
        # Fourier encoding
        proj = 2 * np.pi * torch.matmul(pos, self.B)  # [batch, fourier_dim]
        fourier = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # [batch, 64]

        # Concatenate
        x = torch.cat([pos, fourier], dim=-1)  # [batch, 66]

        # MLP
        h = self.mlp(x)  # [batch, hidden_dim]

        # Generate FiLM parameters
        gamma = self.gamma_head(h)  # [batch, d_model]
        beta = self.beta_head(h)    # [batch, d_model]

        return gamma, beta


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation: h_out = γ ⊙ norm(h) + β

    Magnitude Explosion 해결: dt_7의 60배 증폭을 γ로 제어
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.norm = RMSNorm(d_model)

    def forward(self, h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [batch, seq_len, d_model]
            gamma: [batch, d_model]
            beta: [batch, d_model]
        Returns:
            h_out: [batch, seq_len, d_model]
        """
        h_norm = self.norm(h)

        # Broadcast to sequence dimension
        gamma = gamma.unsqueeze(1)  # [batch, 1, d_model]
        beta = beta.unsqueeze(1)    # [batch, 1, d_model]

        return gamma * h_norm + beta


# =============================================================================
# Spatial MoE Components
# =============================================================================

class SpatialRouter(nn.Module):
    """
    Spatial Routing with Fourier Feature Enhancement

    G(pos) = 1: Boundary Expert
    G(pos) = 0: In-field Expert

    Fourier Features: 6D input → 64D high-frequency projection
    - 100m와 104m 사이의 미세한 경계 변화를 감지
    - cos(2πBx), sin(2πBx) 형태로 고주파 정보 추출

    Hard Switching: Temperature → 0.05 for decisive routing
    """
    def __init__(self, hidden_dim: int = 64, fourier_dim: int = 32, fourier_scale: float = 5.0):
        super().__init__()

        # Fourier Feature Layer for Router
        # Input: 6D (pos_norm[2] + boundary_dists[4])
        # Output: 6 + 2*fourier_dim*6 = 6 + 384 (if fourier_dim=32)
        # Simplified: project to fourier_dim, then concat
        self.fourier_dim = fourier_dim
        self.fourier_scale = fourier_scale

        # Random Fourier projection matrix (fixed)
        # B: [6, fourier_dim] - projects 6D input to fourier_dim frequencies
        B = torch.randn(6, fourier_dim) * fourier_scale
        self.register_buffer('B', B)

        # After Fourier: 6 (raw) + 2*fourier_dim (cos+sin) = 6 + 64 = 70
        fourier_output_dim = 6 + 2 * fourier_dim

        # Router MLP with Fourier-enhanced input
        self.router = nn.Sequential(
            nn.Linear(fourier_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def compute_boundary_distances(self, pos: torch.Tensor) -> torch.Tensor:
        """Compute normalized distances to all 4 boundaries."""
        x, y = pos[:, 0], pos[:, 1]

        y_max_norm = FIELD_Y / FIELD_X
        dist_left = x
        dist_right = 1.0 - x
        dist_bottom = y
        dist_top = y_max_norm - y

        return torch.stack([dist_left, dist_right, dist_bottom, dist_top], dim=-1)

    def fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier feature encoding: x → [x, cos(2πBx), sin(2πBx)]

        Args:
            x: [batch, 6] - normalized spatial features
        Returns:
            [batch, 6 + 2*fourier_dim] - Fourier-enhanced features
        """
        # x @ B: [batch, fourier_dim]
        proj = 2 * np.pi * (x @ self.B)
        fourier_features = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return torch.cat([x, fourier_features], dim=-1)

    def forward(self, pos: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Args:
            pos: [batch, 2] - (start_x_7, start_y_7)
            temperature: temperature for sigmoid (lower = sharper decisions)
        Returns:
            gate: [batch, 1] - boundary expert weight
        """
        boundary_dists = self.compute_boundary_distances(pos)

        # Combine spatial features
        spatial_features = torch.cat([pos, boundary_dists], dim=-1)  # [batch, 6]

        # Apply Fourier encoding for high-frequency sensitivity
        fourier_features = self.fourier_encode(spatial_features)  # [batch, 70]

        # Route through MLP
        logit = self.router(fourier_features) / (temperature + 1e-6)
        gate = torch.sigmoid(logit)

        return gate


class MDNExpertHead(nn.Module):
    """
    Mixture Density Network Expert Head (K=2 modes: In-play, Out-of-play)

    Output per mode:
    - π: mixing coefficient (softmax, sum=1)
    - μ: mean coordinates (x, y)
    - σ: standard deviation (softplus + epsilon)

    Total output: K × (1 + 2 + 2) = 10 dimensions for K=2
    """
    def __init__(self, d_model: int = 128, dropout: float = 0.2,
                 shortcut_dim: int = 0, n_modes: int = 2):
        super().__init__()

        self.shortcut_dim = shortcut_dim
        self.n_modes = n_modes
        self.sigma_min = 0.01  # Minimum sigma for numerical stability

        input_dim = d_model + shortcut_dim

        # Shared representation
        self.shared = nn.Sequential(
            RMSNorm(d_model) if shortcut_dim == 0 else nn.Identity(),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        # If shortcut exists, add normalization before concat
        if shortcut_dim > 0:
            self.cls_norm = RMSNorm(d_model)
            self.shortcut_norm = nn.LayerNorm(shortcut_dim)

        # π head: mixing coefficients [batch, n_modes]
        self.pi_head = nn.Linear(d_model, n_modes)

        # μ head: means [batch, n_modes * 2]
        self.mu_head = nn.Linear(d_model, n_modes * 2)

        # σ head: standard deviations [batch, n_modes * 2]
        self.sigma_head = nn.Linear(d_model, n_modes * 2)

        self._init_weights()

    def _init_weights(self) -> None:
        # Start near field center: x ~ 0.5, y ~ (FIELD_Y/FIELD_X)*0.5
        y_center = 0.5 * (FIELD_Y / FIELD_X)
        nn.init.normal_(self.mu_head.weight, std=0.001)
        with torch.no_grad():
            self.mu_head.bias[: self.n_modes].fill_(0.5)
            self.mu_head.bias[self.n_modes :].fill_(y_center)

        nn.init.normal_(self.sigma_head.weight, std=0.001)
        nn.init.constant_(self.sigma_head.bias, -3.0)

    def forward(self, h: torch.Tensor, shortcut: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            h: CLS token [batch, d_model]
            shortcut: Optional Fourier features [batch, shortcut_dim]
        Returns:
            pi: [batch, n_modes] mixing coefficients (sum=1)
            mu: [batch, n_modes, 2] means
            sigma: [batch, n_modes, 2] standard deviations
        """
        batch_size = h.size(0)

        if self.shortcut_dim > 0 and shortcut is not None:
            h = self.cls_norm(h)
            shortcut = self.shortcut_norm(shortcut)
            h = torch.cat([h, shortcut], dim=-1)

        shared = self.shared(h)

        # π: mixing coefficients with softmax
        pi = F.softmax(self.pi_head(shared), dim=-1)  # [batch, n_modes]

        # μ: means
        mu = self.mu_head(shared)  # [batch, n_modes * 2]
        mu = mu.view(batch_size, self.n_modes, 2)  # [batch, n_modes, 2]

        # σ: standard deviations with softplus + min
        sigma = F.softplus(self.sigma_head(shared)) + self.sigma_min  # [batch, n_modes * 2]
        sigma = sigma.view(batch_size, self.n_modes, 2)  # [batch, n_modes, 2]

        return {'pi': pi, 'mu': mu, 'sigma': sigma}


class AuxiliaryBoundaryHead(nn.Module):
    """
    Auxiliary Task: Boundary Distance + Zone Classification

    모델이 공간 구조를 더 잘 이해하도록 유도
    """
    def __init__(self, d_model: int = 128):
        super().__init__()

        # Distance regression
        self.dist_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

        # Zone classification (4 classes)
        self.zone_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4)
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns: dist_pred [batch, 1], zone_logits [batch, 4]"""
        return self.dist_head(h), self.zone_head(h)


# =============================================================================
# Main Model: FiLM + Spatial MoE Transformer
# =============================================================================

class FiLMSpatialMoETransformer(nn.Module):
    """
    FiLM + Spatial MoE Hybrid Transformer

    1. FourierFeatureLayer: 좌표 인코딩
    2. Transformer + FiLM: 위치 조건부 특성 추출
    3. Spatial MoE: 영역별 전문가 혼합
    4. Heteroscedastic Output: y = μ + σ ⊙ r
    """
    def __init__(
        self,
        num_cont_features: int,
        cat_cardinalities: Tuple[int, ...],
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        fourier_mapping_size: int = 32,
        fourier_scale: float = 5.0,
        film_hidden_dim: int = 64,
        router_hidden_dim: int = 32,
    ):
        super().__init__()

        self.d_model = d_model
        self.K = K
        self.coord_dim = 4

        # Categorical embeddings
        self.embedding_dim = 8
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, self.embedding_dim, padding_idx=0)
            for cardinality in cat_cardinalities
        ])
        total_emb_dim = len(cat_cardinalities) * self.embedding_dim

        # Fourier Feature Layer
        self.fourier_layer = FourierFeatureLayer(
            2, fourier_mapping_size, fourier_scale
        )

        # Total input size
        total_input_size = (
            self.coord_dim + self.fourier_layer.output_dim +
            num_cont_features + total_emb_dim
        )

        # Input projection
        self.input_proj = nn.Linear(total_input_size, d_model)

        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, K + 1, d_model))

        # Transformer layers
        self.layers = nn.ModuleList([
            GatedTransformerBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        # FiLM components
        self.conditioning_net = SpatialConditioningNetwork(d_model, film_hidden_dim)
        self.film_layers = nn.ModuleList([
            FiLMLayer(d_model) for _ in range(num_layers)
        ])

        # Final normalization
        self.final_norm = RMSNorm(d_model)

        # Spatial MoE components with MDN (K=2 modes: In-play, Out-of-play)
        self.n_modes = 2
        self.spatial_router = SpatialRouter(router_hidden_dim)
        self.expert_infield = MDNExpertHead(d_model, dropout, shortcut_dim=0, n_modes=self.n_modes)

        # Expert B gets Fourier shortcut: 6 raw + 64 Fourier = 70
        self.boundary_shortcut_dim = 70
        self.expert_boundary = MDNExpertHead(d_model, dropout,
                                             shortcut_dim=self.boundary_shortcut_dim,
                                             n_modes=self.n_modes)

        # Separate Fourier encoder for boundary shortcut (same as router)
        self.boundary_fourier_dim = 32
        self.boundary_fourier_scale = 5.0
        B = torch.randn(6, self.boundary_fourier_dim) * self.boundary_fourier_scale
        self.register_buffer('boundary_B', B)

        # Auxiliary head
        self.aux_head = AuxiliaryBoundaryHead(d_model)

    def extract_position(self, x_coords: torch.Tensor) -> torch.Tensor:
        """Extract (start_x_7, start_y_7) from input sequence."""
        return x_coords[:, -1, :2]

    def forward(
        self,
        x_coords: torch.Tensor,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        padding_mask: torch.Tensor,
        temperature: float = 1.0,
        return_all: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x_coords: [batch, 8, 4] - normalized coords
            x_cont: [batch, 8, N] - standardized continuous features
            x_cat: [batch, 8, M] - categorical ids
            padding_mask: [batch, 8] - True for padded timesteps
            temperature: router temperature (lower = sharper routing)
            return_all: whether to return all outputs
        """
        batch_size, seq_len, _ = x_coords.shape

        # 1. Extract spatial position
        pos = self.extract_position(x_coords)  # [batch, 2]

        # 2. Generate FiLM parameters
        gamma, beta = self.conditioning_net(pos)

        # 3. Fourier encoding
        pos_seq = x_coords[:, :, :2]
        fourier_features = self.fourier_layer(pos_seq)
        emb_features = [
            emb(x_cat[:, :, i])
            for i, emb in enumerate(self.embeddings)
        ]
        x_emb = torch.cat(emb_features, dim=-1) if emb_features else x_coords.new_zeros(batch_size, seq_len, 0)
        x = torch.cat([x_coords, fourier_features, x_cont, x_emb], dim=-1)

        # 4. Input projection
        x = self.input_proj(x)

        # 5. Add CLS token and position embedding
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]

        # 6. Transformer layers with FiLM
        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
        attn_mask = torch.cat([cls_mask, padding_mask], dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x, src_key_padding_mask=attn_mask)
            x = self.film_layers[i](x, gamma, beta)

        # 7. Final norm and CLS extraction
        x = self.final_norm(x)
        cls_out = x[:, 0, :]

        # 8. Spatial routing with temperature
        gate = self.spatial_router(pos, temperature=temperature)  # [batch, 1]

        # 9. Compute boundary Fourier shortcut + Interaction for Expert B
        # Same encoding as router: pos_norm + boundary_distances + Fourier features
        y_max_norm = FIELD_Y / FIELD_X
        dist_left = pos[:, 0:1]
        dist_right = 1.0 - pos[:, 0:1]
        dist_bottom = pos[:, 1:2]
        dist_top = y_max_norm - pos[:, 1:2]
        raw_features = torch.cat([pos, dist_left, dist_right, dist_bottom, dist_top], dim=-1)  # [batch, 6]

        # Fourier encoding
        proj = 2 * np.pi * (raw_features @ self.boundary_B)
        fourier_shortcut = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)  # [batch, 64]
        boundary_shortcut = torch.cat([raw_features, fourier_shortcut], dim=-1)  # [batch, 70]

        # 10. MDN Expert predictions
        # Expert A: In-field specialist
        mdn_A = self.expert_infield(cls_out)
        # Expert B: Boundary specialist with shortcut
        mdn_B = self.expert_boundary(cls_out, shortcut=boundary_shortcut)

        # Extract MDN parameters: pi [batch, K], mu [batch, K, 2], sigma [batch, K, 2]
        pi_A, mu_A, sigma_A = mdn_A['pi'], mdn_A['mu'], mdn_A['sigma']
        pi_B, mu_B, sigma_B = mdn_B['pi'], mdn_B['mu'], mdn_B['sigma']

        # 11. Mixture of experts via gate
        # gate: [batch, 1] -> expand for mixing
        gate_expanded = gate.unsqueeze(-1)  # [batch, 1, 1]

        # Mixed MDN parameters
        pi_mixed = gate * pi_B + (1 - gate) * pi_A  # [batch, K]
        mu_mixed = gate_expanded * mu_B + (1 - gate_expanded) * mu_A  # [batch, K, 2]
        sigma_mixed = gate_expanded * sigma_B + (1 - gate_expanded) * sigma_A  # [batch, K, 2]

        # 12. Maximum Mode Selection for inference (argmax π)
        # Select the mode with highest mixing coefficient
        max_mode_idx = pi_mixed.argmax(dim=1)  # [batch]
        batch_idx = torch.arange(batch_size, device=pi_mixed.device)

        # y_final = μ of the most probable mode
        y_final = mu_mixed[batch_idx, max_mode_idx]  # [batch, 2]

        # Also compute weighted average for comparison
        # y_avg = Σ π_k * μ_k
        y_avg = (pi_mixed.unsqueeze(-1) * mu_mixed).sum(dim=1)  # [batch, 2]

        # 13. Auxiliary predictions
        aux_dist, aux_zone = self.aux_head(cls_out)

        outputs = {
            # Final predictions
            'y_final': y_final,      # Maximum mode selection
            'y_avg': y_avg,          # Weighted average (for comparison)
            # Mixed MDN parameters
            'pi': pi_mixed,          # [batch, K]
            'mu': mu_mixed,          # [batch, K, 2]
            'sigma': sigma_mixed,    # [batch, K, 2]
            # Expert-specific MDN
            'pi_A': pi_A, 'mu_A': mu_A, 'sigma_A': sigma_A,
            'pi_B': pi_B, 'mu_B': mu_B, 'sigma_B': sigma_B,
            # Router
            'gate': gate,
            # FiLM
            'gamma': gamma,
            'beta': beta,
            # Auxiliary
            'aux_dist': aux_dist,
            'aux_zone': aux_zone,
        }

        return outputs


# =============================================================================
# Loss Function
# =============================================================================

def gmm_nll_loss(y_true: torch.Tensor, pi: torch.Tensor,
                  mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Gaussian Mixture Model Negative Log-Likelihood with Log-Sum-Exp trick.

    Args:
        y_true: [batch, 2] target coordinates
        pi: [batch, K] mixing coefficients (sum=1)
        mu: [batch, K, 2] means
        sigma: [batch, K, 2] standard deviations

    Returns:
        NLL loss (scalar)

    Formula:
        -log( Σ_k π_k * N(y | μ_k, σ_k) )
        = -logsumexp_k( log(π_k) + log(N(y | μ_k, σ_k)) )

    Log of Gaussian:
        log N(y|μ,σ) = -0.5 * ( log(2π) + 2*log(σ) + ((y-μ)/σ)² )
    """
    batch_size, n_modes, _ = mu.shape

    # Expand y_true for broadcasting: [batch, 1, 2] -> [batch, K, 2]
    y_expanded = y_true.unsqueeze(1).expand_as(mu)

    # Gaussian log-likelihood per mode per dimension
    # log N(y|μ,σ) = -0.5 * (log(2π) + 2*log(σ) + ((y-μ)/σ)²)
    log_2pi = np.log(2 * np.pi)
    diff = (y_expanded - mu) / sigma  # [batch, K, 2]
    log_gaussian = -0.5 * (log_2pi + 2 * torch.log(sigma) + diff ** 2)  # [batch, K, 2]

    # Sum over dimensions (x, y are independent)
    log_gaussian = log_gaussian.sum(dim=-1)  # [batch, K]

    # Add log(π_k)
    log_pi = torch.log(pi + 1e-10)  # [batch, K]
    log_weighted = log_pi + log_gaussian  # [batch, K]

    # Log-Sum-Exp for mixture
    log_prob = torch.logsumexp(log_weighted, dim=-1)  # [batch]

    # NLL
    nll = -log_prob.mean()
    return nll


def compute_loss(
    outputs: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    boundary_dist: torch.Tensor,
    boundary_zone: torch.Tensor,
    w_nll: float = 1.0,
    w_final: float = 5.0,
    w_aux_dist: float = 0.3,
    w_aux_zone: float = 0.3,
    w_gate: float = 0.5,
    w_isolation: float = 3.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    MDN-based Multi-task Loss with Hard Expert Isolation.

    핵심:
    1. GMM NLL for multi-modal prediction
    2. Expert A: In-field only, Expert B: Boundary only
    3. Goal-line importance weighting (5x)
    """

    # Masks
    infield_mask = (boundary_zone == 0)
    boundary_mask = (boundary_zone != 0)
    goalline_mask = (boundary_zone == 3)

    # =========================================================================
    # 1. Mixed GMM NLL (전체 모델 학습)
    # =========================================================================
    pi_mixed = outputs['pi']       # [batch, K]
    mu_mixed = outputs['mu']       # [batch, K, 2]
    sigma_mixed = outputs['sigma'] # [batch, K, 2]

    gmm_nll = gmm_nll_loss(y_true, pi_mixed, mu_mixed, sigma_mixed)

    # MSE for final prediction (Maximum Mode Selection)
    mse_final = F.mse_loss(outputs['y_final'], y_true)

    # =========================================================================
    # 2. Hard Expert Isolation with GMM NLL
    # =========================================================================
    pi_A, mu_A, sigma_A = outputs['pi_A'], outputs['mu_A'], outputs['sigma_A']
    pi_B, mu_B, sigma_B = outputs['pi_B'], outputs['mu_B'], outputs['sigma_B']

    # Expert A: In-field only
    if infield_mask.any():
        nll_A = gmm_nll_loss(
            y_true[infield_mask],
            pi_A[infield_mask],
            mu_A[infield_mask],
            sigma_A[infield_mask]
        )
    else:
        nll_A = torch.tensor(0.0, device=y_true.device)

    # Expert B: Boundary only with Goal-line importance weighting
    BOUNDARY_WEIGHT = 3.0
    GOALLINE_WEIGHT = 5.0

    nll_B = torch.tensor(0.0, device=y_true.device)

    # Non-goalline boundary (Top-out, Bottom-out)
    sideline_mask = boundary_mask & ~goalline_mask
    if sideline_mask.any():
        nll_B_side = gmm_nll_loss(
            y_true[sideline_mask],
            pi_B[sideline_mask],
            mu_B[sideline_mask],
            sigma_B[sideline_mask]
        )
        nll_B = nll_B + BOUNDARY_WEIGHT * nll_B_side

    # Goal-line (highest weight)
    if goalline_mask.any():
        nll_B_gl = gmm_nll_loss(
            y_true[goalline_mask],
            pi_B[goalline_mask],
            mu_B[goalline_mask],
            sigma_B[goalline_mask]
        )
        nll_B = nll_B + GOALLINE_WEIGHT * nll_B_gl

    isolation_loss = nll_A + nll_B

    # =========================================================================
    # 3. Auxiliary Losses
    # =========================================================================
    aux_dist_loss = F.smooth_l1_loss(outputs['aux_dist'], boundary_dist)
    aux_zone_loss = F.cross_entropy(outputs['aux_zone'], boundary_zone, label_smoothing=0.1)

    # =========================================================================
    # 4. Gate Supervision
    # =========================================================================
    gate = outputs['gate']
    gate_target = boundary_mask.float().unsqueeze(1)
    gate_loss = F.binary_cross_entropy(gate, gate_target)

    # =========================================================================
    # 5. Mode Distribution 모니터링
    # =========================================================================
    with torch.no_grad():
        # Which mode is selected most often?
        max_mode = pi_mixed.argmax(dim=1)  # [batch]
        mode0_ratio = (max_mode == 0).float().mean().item()
        mode1_ratio = (max_mode == 1).float().mean().item()

        # Distance comparison: y_final (max mode) vs y_avg (weighted avg)
        dist_final = torch.sqrt(((outputs['y_final'] - y_true) ** 2).sum(dim=1)).mean().item()
        dist_avg = torch.sqrt(((outputs['y_avg'] - y_true) ** 2).sum(dim=1)).mean().item()

    # =========================================================================
    # 6. Total Loss
    # =========================================================================
    total_loss = (
        w_nll * gmm_nll +
        w_final * mse_final +
        w_isolation * isolation_loss +
        w_aux_dist * aux_dist_loss +
        w_aux_zone * aux_zone_loss +
        w_gate * gate_loss
    )

    loss_dict = {
        'gmm_nll': gmm_nll.item(),
        'mse_final': mse_final.item(),
        'nll_A': nll_A.item() if isinstance(nll_A, torch.Tensor) else nll_A,
        'nll_B': nll_B.item() if isinstance(nll_B, torch.Tensor) else nll_B,
        'isolation': isolation_loss.item(),
        'gate': gate_loss.item(),
        'mode0_ratio': mode0_ratio,
        'mode1_ratio': mode1_ratio,
        'dist_final': dist_final,
        'dist_avg': dist_avg,
        'total': total_loss.item(),
    }

    return total_loss, loss_dict


# =============================================================================
# Training Scheduler
# =============================================================================

class TrainingScheduler:
    """
    Simplified Training Schedule with Temperature Annealing

    Phase 1 (Warm-up, 10 epochs): High temperature (soft routing), gate supervision
    Phase 2 (Main, 10-100 epochs): Temperature annealing 2.0 → 0.5
    Phase 3 (Fine-tune, 100+ epochs): Low temperature (sharp routing)

    Gate Supervision is always on to prevent router collapse.
    """
    def __init__(self, warmup_epochs=10, anneal_epochs=90):
        self.warmup_epochs = warmup_epochs
        self.anneal_epochs = anneal_epochs
        self.anneal_end = warmup_epochs + anneal_epochs

    def get_temperature(self, epoch: int) -> float:
        """
        Temperature annealing: 2.0 → 0.05 (Hard Switching)

        High temp = soft decisions (early training)
        Low temp = hard decisions (late training, "A or B")
        """
        if epoch < self.warmup_epochs:
            return 2.0  # Soft during warm-up
        elif epoch < self.anneal_end:
            progress = (epoch - self.warmup_epochs) / self.anneal_epochs
            # 2.0 → 0.05 (hard switching)
            return 2.0 * (0.05 / 2.0) ** progress  # Exponential decay
        else:
            return 0.05  # Hard switching: decisive "A or B"

    def get_loss_weights(self, epoch: int) -> Dict[str, float]:
        """
        Loss weights with Expert Isolation.

        - w_gate: 감소 (라우터가 이미 공간 인식 중)
        - w_isolation: 전문가 격리 학습에 집중
        """
        if epoch < self.warmup_epochs:
            return {
                'w_nll': 1.0, 'w_final': 5.0,
                'w_aux_dist': 0.3, 'w_aux_zone': 0.3,
                'w_gate': 1.0,  # Gate supervision (reduced)
                'w_isolation': 3.0  # Strong expert isolation early
            }
        elif epoch < self.anneal_end:
            progress = (epoch - self.warmup_epochs) / self.anneal_epochs
            return {
                'w_nll': 1.0, 'w_final': 5.0,
                'w_aux_dist': 0.3 + 0.2 * progress,
                'w_aux_zone': 0.3 + 0.2 * progress,
                'w_gate': 1.0 - 0.5 * progress,  # 1.0 → 0.5
                'w_isolation': 3.0  # Maintain isolation
            }
        else:
            return {
                'w_nll': 1.0, 'w_final': 5.0,
                'w_aux_dist': 0.5, 'w_aux_zone': 0.5,
                'w_gate': 0.5,
                'w_isolation': 3.0
            }


# =============================================================================
# Data Preparation
# =============================================================================

def create_boundary_labels(end_x, end_y):
    """Zone labels: 0=In-field, 1=Top-out, 2=Bottom-out, 3=Goal-line"""
    labels = np.zeros(len(end_x), dtype=int)
    labels[(end_x > FIELD_X - 5) | (end_x < 5)] = 3
    labels[(labels == 0) & (end_y > FIELD_Y - 5)] = 1
    labels[(labels == 0) & (end_y < 5)] = 2
    return labels


def compute_boundary_distance(end_x, end_y):
    """Minimum distance to nearest boundary (normalized)."""
    dist_left = end_x
    dist_right = FIELD_X - end_x
    dist_bottom = end_y
    dist_top = FIELD_Y - end_y
    min_dist = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_bottom, dist_top))
    return min_dist / (FIELD_X / 2)  # Normalize


def prepare_sequence_data(df, k=K):
    n_samples = len(df)
    coord_dim = 4
    cont_dim = len(CONT_FEATURES)
    angle_dim = len(ANGLE_FEATURES) * 2
    cat_dim = len(CAT_FEATURES)

    coords_seq = np.full((n_samples, k, coord_dim), -1.0)
    cont_seq = np.zeros((n_samples, k, cont_dim))
    angle_seq = np.zeros((n_samples, k, angle_dim))
    cat_seq = np.zeros((n_samples, k, cat_dim), dtype=int)
    valid_mask_seq = np.zeros((n_samples, k), dtype=bool)
    cont_valid_mask = np.zeros((n_samples, k, cont_dim), dtype=bool)

    coord_idx = {'start_x': 0, 'start_y': 1, 'end_x': 2, 'end_y': 3}
    cont_idx = {feat: i for i, feat in enumerate(CONT_FEATURES)}
    angle_idx = {feat: i for i, feat in enumerate(ANGLE_FEATURES)}
    cat_idx = {feat: i for i, feat in enumerate(CAT_FEATURES)}

    for t in range(k):
        check_col = f'start_x_{t}'
        if check_col not in df.columns:
            continue

        has_data = ~df[check_col].isna()
        valid_mask_seq[has_data, t] = True

        for feat in COORD_FEATURES:
            col = f'{feat}_{t}'
            if col in df.columns and (feat in COORD_FEATURES[:2] or t < k - 1):
                vals = df.loc[has_data, col].values / MAX_LEN
                coords_seq[has_data, t, coord_idx[feat]] = vals

        for feat in CONT_FEATURES:
            col = f'{feat}_{t}'
            if col in df.columns and (feat in CONT_FEATURES[:4] or t < k - 1):
                cont_seq[has_data, t, cont_idx[feat]] = df.loc[has_data, col].values
                cont_valid_mask[has_data, t, cont_idx[feat]] = True

        for feat in ANGLE_FEATURES:
            col = f'{feat}_{t}'
            if col in df.columns and (feat in ('angle_to_goal', 'angle_visible') or t < k - 1):
                rads = df.loc[has_data, col].values
                if feat == 'angle_to_goal':
                    rads = np.deg2rad(rads)
                idx = angle_idx[feat]
                angle_seq[has_data, t, idx * 2] = np.sin(rads)
                angle_seq[has_data, t, idx * 2 + 1] = np.cos(rads)

        for feat in CAT_FEATURES:
            col = f'{feat}_{t}'
            if col in df.columns:
                cat_seq[has_data, t, cat_idx[feat]] = df.loc[has_data, col].astype(int).values + 1

    return coords_seq, cont_seq, angle_seq, cat_seq, valid_mask_seq, cont_valid_mask, cont_idx, angle_idx, cat_idx


def augment_data(coords, cont, angles, cat, padding_mask, cont_valid_mask,
                 y_coords, boundary_dist, boundary_zone, cont_idx, angle_idx, cat_idx):
    """
    Y-flip augmentation only (SMOTE removed - corrupts backbone).
    """
    coords_all = [coords.copy()]
    cont_all = [cont.copy()]
    angles_all = [angles.copy()]
    cat_all = [cat.copy()]
    mask_all = [padding_mask.copy()]
    cont_valid_all = [cont_valid_mask.copy()]
    y_coords_all = [y_coords.copy()]
    boundary_dist_all = [boundary_dist.copy()]
    boundary_zone_all = [boundary_zone.copy()]

    # Y-flip augmentation
    coords_yflip = coords.copy()
    y_max_norm = FIELD_Y / MAX_LEN
    valid_mask = ~padding_mask
    coords_yflip[:, :, 1] = np.where(valid_mask, y_max_norm - coords_yflip[:, :, 1], coords_yflip[:, :, 1])
    coords_yflip[:, :, 3] = np.where(valid_mask, y_max_norm - coords_yflip[:, :, 3], coords_yflip[:, :, 3])

    cont_yflip = cont.copy()
    if 'dy' in cont_idx:
        cont_yflip[:, :, cont_idx['dy']] = -cont_yflip[:, :, cont_idx['dy']]
    if 'action_lateral' in cont_idx:
        cont_yflip[:, :, cont_idx['action_lateral']] = -cont_yflip[:, :, cont_idx['action_lateral']]

    angles_yflip = angles.copy()
    for feat in ANGLE_FLIP_FEATURES:
        if feat in angle_idx:
            idx = angle_idx[feat] * 2
            angles_yflip[:, :, idx] = -angles_yflip[:, :, idx]

    cat_yflip = cat.copy()
    if 'lane' in cat_idx:
        lane = cat_yflip[:, :, cat_idx['lane']].copy()
        cat_yflip[:, :, cat_idx['lane']] = np.where(lane == 0, 2, np.where(lane == 2, 0, lane))

    y_coords_yflip = y_coords.copy()
    y_coords_yflip[:, 1] = FIELD_Y - y_coords_yflip[:, 1]

    zone_yflip = boundary_zone.copy()
    zone_yflip = np.where(zone_yflip == 1, 2, np.where(zone_yflip == 2, 1, zone_yflip))

    coords_all.append(coords_yflip)
    cont_all.append(cont_yflip)
    angles_all.append(angles_yflip)
    cat_all.append(cat_yflip)
    mask_all.append(padding_mask.copy())
    cont_valid_all.append(cont_valid_mask.copy())
    y_coords_all.append(y_coords_yflip)
    boundary_dist_all.append(boundary_dist.copy())
    boundary_zone_all.append(zone_yflip)

    return (np.concatenate(coords_all), np.concatenate(cont_all), np.concatenate(angles_all),
            np.concatenate(cat_all), np.concatenate(mask_all), np.concatenate(cont_valid_all),
            np.concatenate(y_coords_all),
            np.concatenate(boundary_dist_all), np.concatenate(boundary_zone_all))


# =============================================================================
# Main Training
# =============================================================================

def main():
    log("=" * 70)
    log("FiLM + MDN Spatial MoE Transformer")
    log("  - MDN: K=2 modes (In-play, Out-of-play)")
    log("  - FiLM: Magnitude Explosion 해결")
    log("  - Spatial MoE: Sign Reversal 해결")
    log("  - GMM NLL: Multi-modal prediction")
    log("=" * 70)

    # ==========================================================================
    # Data Loading
    # ==========================================================================
    log("\n[1] 데이터 로드...")
    df_all = pd.read_csv(DATA_DIR / 'train_features_v2.csv')
    (coords_seq, cont_seq, angles_seq, cat_seq, valid_mask_seq,
     cont_valid_mask_seq, cont_idx, angle_idx, cat_idx) = prepare_sequence_data(df_all, K)
    y_coords = df_all[['target_end_x', 'target_end_y']].values

    # Auxiliary targets
    boundary_zone = create_boundary_labels(df_all['target_end_x'].values, df_all['target_end_y'].values)
    boundary_dist = compute_boundary_distance(df_all['target_end_x'].values, df_all['target_end_y'].values)

    # Train/Val split
    train_idx, val_idx = train_test_split(range(len(df_all)), test_size=0.2, random_state=42)
    coords_train_orig = coords_seq[train_idx]
    cont_train_orig = cont_seq[train_idx]
    angles_train_orig = angles_seq[train_idx]
    cat_train_orig = cat_seq[train_idx]
    valid_mask_train_orig = valid_mask_seq[train_idx]
    cont_valid_mask_train_orig = cont_valid_mask_seq[train_idx]

    coords_val = coords_seq[val_idx]
    cont_val = cont_seq[val_idx]
    angles_val = angles_seq[val_idx]
    cat_val = cat_seq[val_idx]
    valid_mask_val = valid_mask_seq[val_idx]
    cont_valid_mask_val = cont_valid_mask_seq[val_idx]
    y_coords_train_orig = y_coords[train_idx]
    y_coords_val = y_coords[val_idx]
    boundary_dist_train_orig = boundary_dist[train_idx]
    boundary_dist_val = boundary_dist[val_idx]
    boundary_zone_train_orig = boundary_zone[train_idx]
    boundary_zone_val = boundary_zone[val_idx]

    log(f"  Train: {len(coords_train_orig)}, Val: {len(coords_val)}")
    log(f"  Zone distribution (Val): {np.bincount(boundary_zone_val)}")

    # ==========================================================================
    # Augmentation
    # ==========================================================================
    log("\n[2] 데이터 증강 (Y-flip, 2x)...")
    padding_mask_train_orig = ~valid_mask_train_orig
    padding_mask_val = ~valid_mask_val

    (coords_train, cont_train, angles_train, cat_train,
     padding_mask_train, cont_valid_mask_train,
     y_coords_train, boundary_dist_train, boundary_zone_train) = augment_data(
        coords_train_orig, cont_train_orig, angles_train_orig, cat_train_orig,
        padding_mask_train_orig, cont_valid_mask_train_orig,
        y_coords_train_orig, boundary_dist_train_orig, boundary_zone_train_orig,
        cont_idx, angle_idx, cat_idx
    )
    log(f"  Train (augmented): {len(coords_train)}")
    log(f"  Zone distribution (Train): {np.bincount(boundary_zone_train)}")

    # ==========================================================================
    # Normalization
    # ==========================================================================
    n_samples, n_steps, cont_dim = cont_train.shape
    flat_cont = cont_train.reshape(-1, cont_dim)
    flat_mask = cont_valid_mask_train.reshape(-1, cont_dim)
    counts = flat_mask.sum(axis=0)
    counts = np.where(counts == 0, 1, counts)
    cont_mean = (flat_cont * flat_mask).sum(axis=0) / counts
    cont_var = ((flat_cont - cont_mean) ** 2 * flat_mask).sum(axis=0) / counts
    cont_std = np.sqrt(cont_var)
    cont_std = np.where(cont_std < 1e-6, 1.0, cont_std)

    cont_train = (cont_train - cont_mean) / cont_std
    cont_train[~cont_valid_mask_train] = 0.0
    cont_val = (cont_val - cont_mean) / cont_std
    cont_val[~cont_valid_mask_val] = 0.0

    cont_train = np.concatenate([cont_train, angles_train], axis=-1)
    cont_val = np.concatenate([cont_val, angles_val], axis=-1)

    coord_scale = np.array([MAX_LEN, MAX_LEN])
    y_coords_train_norm = y_coords_train / coord_scale
    y_coords_val_norm = y_coords_val / coord_scale

    cat_cardinalities = tuple(
        int(cat_train_orig[:, :, i].max()) + 1 for i in range(cat_train_orig.shape[2])
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"  Device: {device}")

    # ==========================================================================
    # Create Tensors
    # ==========================================================================
    coords_train_t = torch.FloatTensor(coords_train).to(device)
    cont_train_t = torch.FloatTensor(cont_train).to(device)
    cat_train_t = torch.LongTensor(cat_train).to(device)
    padding_mask_train_t = torch.BoolTensor(padding_mask_train).to(device)
    y_train_t = torch.FloatTensor(y_coords_train_norm).to(device)
    boundary_dist_train_t = torch.FloatTensor(boundary_dist_train.reshape(-1, 1)).to(device)
    boundary_zone_train_t = torch.LongTensor(boundary_zone_train).to(device)

    coords_val_t = torch.FloatTensor(coords_val).to(device)
    cont_val_t = torch.FloatTensor(cont_val).to(device)
    cat_val_t = torch.LongTensor(cat_val).to(device)
    padding_mask_val_t = torch.BoolTensor(padding_mask_val).to(device)
    y_val_t = torch.FloatTensor(y_coords_val_norm).to(device)
    boundary_dist_val_t = torch.FloatTensor(boundary_dist_val.reshape(-1, 1)).to(device)
    boundary_zone_val_t = torch.LongTensor(boundary_zone_val).to(device)

    train_dataset = TensorDataset(
        coords_train_t, cont_train_t, cat_train_t, padding_mask_train_t,
        y_train_t, boundary_dist_train_t, boundary_zone_train_t
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # ==========================================================================
    # Model
    # ==========================================================================
    log("\n[3] FiLM + Spatial MoE Transformer 학습...")

    full_cont_dim = cont_train.shape[2]

    model = FiLMSpatialMoETransformer(
        num_cont_features=full_cont_dim,
        cat_cardinalities=cat_cardinalities,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.2,
        fourier_mapping_size=32,
        fourier_scale=5.0,
        film_hidden_dim=64,
        router_hidden_dim=32,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    training_scheduler = TrainingScheduler(warmup_epochs=10, anneal_epochs=90)

    best_dist = float('inf')
    best_state = None
    patience, patience_counter = 50, 0  # Increased patience

    label_names = ['In-field', 'Top-out', 'Bottom-out', 'Goal-line']

    # ==========================================================================
    # Training Loop
    # ==========================================================================
    for epoch in range(200):
        model.train()

        # Get training schedule parameters
        temperature = training_scheduler.get_temperature(epoch)
        loss_weights = training_scheduler.get_loss_weights(epoch)

        total_losses = {'nll': 0, 'mse_final': 0, 'aux_dist': 0, 'aux_zone': 0, 'gate': 0}

        for coords_batch, cont_batch, cat_batch, mask_batch, y_batch, dist_batch, zone_batch in train_loader:
            optimizer.zero_grad()

            outputs = model(
                coords_batch, cont_batch, cat_batch, mask_batch,
                temperature=temperature, return_all=True
            )

            loss, loss_dict = compute_loss(
                outputs, y_batch, dist_batch, zone_batch, **loss_weights
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in total_losses:
                if k in loss_dict:
                    total_losses[k] += loss_dict[k]

        # Validation (use low temperature for sharp routing)
        model.eval()
        with torch.no_grad():
            outputs = model(
                coords_val_t, cont_val_t, cat_val_t, padding_mask_val_t,
                temperature=0.05, return_all=True
            )

            # Distances (MDN: use maximum mode selection result)
            final_np = outputs['y_final'].cpu().numpy() * coord_scale
            avg_np = outputs['y_avg'].cpu().numpy() * coord_scale

            final_dist = np.sqrt(((final_np - y_coords_val) ** 2).sum(axis=1)).mean()
            avg_dist = np.sqrt(((avg_np - y_coords_val) ** 2).sum(axis=1)).mean()

            # Gate statistics
            gate_np = outputs['gate'].cpu().numpy()
            gate_mean = gate_np.mean()
            gate_std = gate_np.std()

            # Mode distribution (which mode is selected most often)
            pi_np = outputs['pi'].cpu().numpy()  # [batch, K]
            mode_selection = pi_np.argmax(axis=1)
            mode0_ratio = (mode_selection == 0).mean()
            mode1_ratio = (mode_selection == 1).mean()

            # Expert Advantage: B가 A보다 좋은 비율 (Boundary 영역)
            # For MDN, compare best mode of each expert
            mu_A_np = outputs['mu_A'].cpu().numpy()  # [batch, K, 2]
            mu_B_np = outputs['mu_B'].cpu().numpy()
            pi_A_np = outputs['pi_A'].cpu().numpy()
            pi_B_np = outputs['pi_B'].cpu().numpy()

            # Select best mode per expert
            best_mode_A = pi_A_np.argmax(axis=1)
            best_mode_B = pi_B_np.argmax(axis=1)
            batch_idx = np.arange(len(y_coords_val))

            mu_A_best = mu_A_np[batch_idx, best_mode_A] * coord_scale
            mu_B_best = mu_B_np[batch_idx, best_mode_B] * coord_scale

            boundary_mask = boundary_zone_val != 0
            if boundary_mask.sum() > 0:
                mse_A_boundary = ((mu_A_best[boundary_mask] - y_coords_val[boundary_mask]) ** 2).sum(axis=1)
                mse_B_boundary = ((mu_B_best[boundary_mask] - y_coords_val[boundary_mask]) ** 2).sum(axis=1)
                b_better = (mse_B_boundary < mse_A_boundary).mean()
            else:
                b_better = 0.0

            # Auxiliary accuracy
            zone_pred = outputs['aux_zone'].argmax(dim=1).cpu().numpy()
            zone_acc = (zone_pred == boundary_zone_val).mean()

        lr_scheduler.step(final_dist)

        if final_dist < best_dist:
            best_dist = final_dist
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            n_batches = len(train_loader)
            phase = "Warm-up" if epoch < 10 else ("Anneal" if epoch < 100 else "Fine-tune")
            log(f"  Epoch {epoch+1:3d} [{phase:8s}] T={temperature:.2f}: "
                f"Final={final_dist:.2f}m, Mode0={mode0_ratio:.1%}, "
                f"Gate={gate_mean:.3f}±{gate_std:.3f}")

        if patience_counter >= patience:
            log(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    log(f"\n  Best Final Distance: {best_dist:.4f}m")

    # ==========================================================================
    # Final Evaluation
    # ==========================================================================
    log("\n[4] 최종 평가...")
    model.eval()
    with torch.no_grad():
        outputs = model(
            coords_val_t, cont_val_t, cat_val_t, padding_mask_val_t,
            temperature=0.05, return_all=True
        )

        # MDN outputs
        pi_np = outputs['pi'].cpu().numpy()  # [batch, K]
        mu_np = outputs['mu'].cpu().numpy()  # [batch, K, 2]
        sigma_np = outputs['sigma'].cpu().numpy()  # [batch, K, 2]
        final_np = outputs['y_final'].cpu().numpy() * coord_scale  # Max mode
        avg_np = outputs['y_avg'].cpu().numpy() * coord_scale  # Weighted avg
        gate_np = outputs['gate'].cpu().numpy()
        gamma_np = outputs['gamma'].cpu().numpy()
        beta_np = outputs['beta'].cpu().numpy()

    # Select best mode for each sample
    best_mode = pi_np.argmax(axis=1)
    batch_idx = np.arange(len(y_coords_val))
    mu_best = mu_np[batch_idx, best_mode] * coord_scale  # [batch, 2]

    # Overall metrics
    best_mode_dist = np.sqrt(((mu_best - y_coords_val) ** 2).sum(axis=1)).mean()
    final_dist = np.sqrt(((final_np - y_coords_val) ** 2).sum(axis=1)).mean()
    avg_dist = np.sqrt(((avg_np - y_coords_val) ** 2).sum(axis=1)).mean()

    log("\n" + "=" * 70)
    log("[MDN Regression Performance]")
    log("=" * 70)
    log(f"  Max Mode (argmax π) Distance: {final_dist:.4f}m")
    log(f"  Weighted Avg (Σπμ) Distance:  {avg_dist:.4f}m")
    log(f"  Improvement (Avg→Max):        {avg_dist - final_dist:.4f}m")

    # Mode distribution
    log("\n" + "=" * 70)
    log("[Mode Distribution]")
    log("=" * 70)
    mode0_ratio = (best_mode == 0).mean()
    mode1_ratio = (best_mode == 1).mean()
    log(f"  Mode 0 (In-play): {mode0_ratio:.1%}")
    log(f"  Mode 1 (Out-of-play): {mode1_ratio:.1%}")

    # Zone-wise performance
    log("\n" + "=" * 70)
    log("[Zone-wise Performance]")
    log("=" * 70)
    log(f"  {'Zone':12s} | {'N':>5s} | {'MaxMode':>8s} | {'WgtAvg':>8s} | {'Gate':>6s} | {'Mode1%':>7s}")
    log("-" * 65)

    for i, name in enumerate(label_names):
        mask = boundary_zone_val == i
        if mask.sum() > 0:
            max_d = np.sqrt(((final_np[mask] - y_coords_val[mask]) ** 2).sum(axis=1)).mean()
            avg_d = np.sqrt(((avg_np[mask] - y_coords_val[mask]) ** 2).sum(axis=1)).mean()
            avg_gate = gate_np[mask].mean()
            mode1_pct = (best_mode[mask] == 1).mean()
            log(f"  {name:12s} | {mask.sum():5d} | {max_d:8.2f} | {avg_d:8.2f} | {avg_gate:6.3f} | {mode1_pct:6.1%}")

    # FiLM statistics
    log("\n" + "=" * 70)
    log("[FiLM Statistics]")
    log("=" * 70)
    log(f"  γ: mean={gamma_np.mean():.4f}, std={gamma_np.std():.4f}")
    log(f"  β: mean={beta_np.mean():.4f}, std={beta_np.std():.4f}")

    # Gate distribution by zone
    log("\n" + "=" * 70)
    log("[Gate Distribution by Zone]")
    log("=" * 70)
    for i, name in enumerate(label_names):
        mask = boundary_zone_val == i
        if mask.sum() > 0:
            g = gate_np[mask]
            log(f"  {name:12s}: mean={g.mean():.3f}, std={g.std():.3f}, min={g.min():.3f}, max={g.max():.3f}")

    # Mode separation analysis (핵심!)
    log("\n" + "=" * 70)
    log("[Mode Separation Analysis]")
    log("=" * 70)
    # For Goal-line samples, check if modes are separated
    goalline_mask = boundary_zone_val == 3
    if goalline_mask.sum() > 0:
        mu_gl = mu_np[goalline_mask] * coord_scale[0]  # Scale x coord
        mode_diff = np.abs(mu_gl[:, 0, 0] - mu_gl[:, 1, 0]).mean()  # Diff between mode 0 and 1 x-coords
        log(f"  Goal-line Mode Separation (x): {mode_diff:.2f}m")
        log(f"  (Expected: ~5m if modes are In-play vs Out-of-play)")

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'cont_mean': cont_mean,
        'cont_std': cont_std,
        'coord_scale': coord_scale,
        'cat_cardinalities': cat_cardinalities,
        'best_dist': best_dist
    }, DATA_DIR / 'film_smoe_transformer.pt')
    log(f"\n저장: {DATA_DIR / 'film_smoe_transformer.pt'}")

    log("\n" + "=" * 70)
    log("완료!")
    log("=" * 70)


if __name__ == '__main__':
    main()
