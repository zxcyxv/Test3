from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FIELD_X, FIELD_Y, K


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
            nn.Dropout(dropout),
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
        self.register_buffer("B", B)
        self.output_dim = mapping_size * 2

    def forward(self, x):
        x_proj = 2 * np.pi * torch.matmul(x, self.B)
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class SpatialConditioningNetwork(nn.Module):
    def __init__(self, d_model: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.d_model = d_model

        self.fourier_dim = 32
        torch.manual_seed(43)
        B = torch.randn(2, self.fourier_dim) * 5.0
        self.register_buffer("B", B)

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
        proj = 2 * np.pi * torch.matmul(pos, self.B)
        fourier = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        x = torch.cat([pos, fourier], dim=-1)
        h = self.mlp(x)
        gamma = self.gamma_head(h)
        beta = self.beta_head(h)
        return gamma, beta


class FiLMLayer(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.norm = RMSNorm(d_model)

    def forward(self, h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm(h)
        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)
        return gamma * h_norm + beta


class SpatialRouter(nn.Module):
    def __init__(self, hidden_dim: int = 64, fourier_dim: int = 32, fourier_scale: float = 5.0):
        super().__init__()
        self.fourier_dim = fourier_dim
        self.fourier_scale = fourier_scale
        B = torch.randn(6, fourier_dim) * fourier_scale
        self.register_buffer("B", B)

        fourier_output_dim = 6 + 2 * fourier_dim
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
        x, y = pos[:, 0], pos[:, 1]
        y_max_norm = FIELD_Y / FIELD_X
        dist_left = x
        dist_right = 1.0 - x
        dist_bottom = y
        dist_top = y_max_norm - y
        return torch.stack([dist_left, dist_right, dist_bottom, dist_top], dim=-1)

    def fourier_encode(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * np.pi * (x @ self.B)
        fourier_features = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        return torch.cat([x, fourier_features], dim=-1)

    def forward(self, pos: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        boundary_dists = self.compute_boundary_distances(pos)
        spatial_features = torch.cat([pos, boundary_dists], dim=-1)
        fourier_features = self.fourier_encode(spatial_features)
        logit = self.router(fourier_features) / (temperature + 1e-6)
        return torch.sigmoid(logit)


class MDNExpertHead(nn.Module):
    def __init__(self, d_model: int = 128, dropout: float = 0.2,
                 shortcut_dim: int = 0, n_modes: int = 2):
        super().__init__()
        self.shortcut_dim = shortcut_dim
        self.n_modes = n_modes
        self.sigma_min = 0.1
        input_dim = d_model + shortcut_dim

        self.shared = nn.Sequential(
            RMSNorm(d_model) if shortcut_dim == 0 else nn.Identity(),
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )

        if shortcut_dim > 0:
            self.cls_norm = RMSNorm(d_model)
            self.shortcut_norm = nn.LayerNorm(shortcut_dim)

        self.pi_head = nn.Linear(d_model, n_modes)
        self.mu_head = nn.Linear(d_model, n_modes * 2)
        self.sigma_head = nn.Linear(d_model, n_modes * 2)

    def forward(self, h: torch.Tensor, shortcut: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        batch_size = h.size(0)
        if self.shortcut_dim > 0 and shortcut is not None:
            h = self.cls_norm(h)
            shortcut = self.shortcut_norm(shortcut)
            h = torch.cat([h, shortcut], dim=-1)

        shared = self.shared(h)
        pi = F.softmax(self.pi_head(shared), dim=-1)
        mu = self.mu_head(shared).view(batch_size, self.n_modes, 2)
        sigma = F.softplus(self.sigma_head(shared)) + self.sigma_min
        sigma = sigma.view(batch_size, self.n_modes, 2)
        return {"pi": pi, "mu": mu, "sigma": sigma}


class AuxiliaryBoundaryHead(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.dist_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )
        self.zone_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4),
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.dist_head(h), self.zone_head(h)


class FiLMSpatialMoETransformer(nn.Module):
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

        self.embedding_dim = 8
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, self.embedding_dim, padding_idx=0)
            for cardinality in cat_cardinalities
        ])
        total_emb_dim = len(cat_cardinalities) * self.embedding_dim

        self.fourier_layer = FourierFeatureLayer(
            2, fourier_mapping_size, fourier_scale
        )

        total_input_size = (
            self.coord_dim + self.fourier_layer.output_dim +
            num_cont_features + total_emb_dim
        )

        self.input_proj = nn.Linear(total_input_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, K + 1, d_model))

        self.layers = nn.ModuleList([
            GatedTransformerBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])

        self.conditioning_net = SpatialConditioningNetwork(d_model, film_hidden_dim)
        self.film_layers = nn.ModuleList([FiLMLayer(d_model) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model)

        self.n_modes = 2
        self.spatial_router = SpatialRouter(router_hidden_dim)
        self.expert_infield = MDNExpertHead(d_model, dropout, shortcut_dim=0, n_modes=self.n_modes)
        self.boundary_shortcut_dim = 70
        self.expert_boundary = MDNExpertHead(
            d_model, dropout, shortcut_dim=self.boundary_shortcut_dim, n_modes=self.n_modes
        )

        self.boundary_fourier_dim = 32
        self.boundary_fourier_scale = 5.0
        B = torch.randn(6, self.boundary_fourier_dim) * self.boundary_fourier_scale
        self.register_buffer("boundary_B", B)

        self.aux_head = AuxiliaryBoundaryHead(d_model)

    def extract_position(self, x_coords: torch.Tensor) -> torch.Tensor:
        return x_coords[:, -1, :2]

    def forward(
        self,
        x_coords: torch.Tensor,
        x_cont: torch.Tensor,
        x_cat: torch.Tensor,
        padding_mask: torch.Tensor,
        temperature: float = 1.0,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = x_coords.shape
        pos = self.extract_position(x_coords)
        gamma, beta = self.conditioning_net(pos)

        pos_seq = x_coords[:, :, :2]
        fourier_features = self.fourier_layer(pos_seq)
        emb_features = [emb(x_cat[:, :, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(emb_features, dim=-1) if emb_features else x_coords.new_zeros(batch_size, seq_len, 0)
        x = torch.cat([x_coords, fourier_features, x_cont, x_emb], dim=-1)

        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]

        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
        attn_mask = torch.cat([cls_mask, padding_mask], dim=1)
        for i, layer in enumerate(self.layers):
            x = layer(x, src_key_padding_mask=attn_mask)
            x = self.film_layers[i](x, gamma, beta)

        x = self.final_norm(x)
        cls_out = x[:, 0, :]

        gate = self.spatial_router(pos, temperature=temperature)

        y_max_norm = FIELD_Y / FIELD_X
        dist_left = pos[:, 0:1]
        dist_right = 1.0 - pos[:, 0:1]
        dist_bottom = pos[:, 1:2]
        dist_top = y_max_norm - pos[:, 1:2]
        raw_features = torch.cat([pos, dist_left, dist_right, dist_bottom, dist_top], dim=-1)

        proj = 2 * np.pi * (raw_features @ self.boundary_B)
        fourier_shortcut = torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)
        boundary_shortcut = torch.cat([raw_features, fourier_shortcut], dim=-1)

        mdn_A = self.expert_infield(cls_out)
        mdn_B = self.expert_boundary(cls_out, shortcut=boundary_shortcut)

        pi_A, mu_A, sigma_A = mdn_A["pi"], mdn_A["mu"], mdn_A["sigma"]
        pi_B, mu_B, sigma_B = mdn_B["pi"], mdn_B["mu"], mdn_B["sigma"]

        gate_expanded = gate.unsqueeze(-1)
        pi_mixed = gate * pi_B + (1 - gate) * pi_A
        mu_mixed = gate_expanded * mu_B + (1 - gate_expanded) * mu_A
        sigma_mixed = gate_expanded * sigma_B + (1 - gate_expanded) * sigma_A

        max_mode_idx = pi_mixed.argmax(dim=1)
        batch_idx = torch.arange(batch_size, device=pi_mixed.device)
        y_final = mu_mixed[batch_idx, max_mode_idx]
        y_avg = (pi_mixed.unsqueeze(-1) * mu_mixed).sum(dim=1)

        aux_dist, aux_zone = self.aux_head(cls_out)

        return {
            "y_final": y_final,
            "y_avg": y_avg,
            "pi": pi_mixed,
            "mu": mu_mixed,
            "sigma": sigma_mixed,
            "pi_A": pi_A,
            "mu_A": mu_A,
            "sigma_A": sigma_A,
            "pi_B": pi_B,
            "mu_B": mu_B,
            "sigma_B": sigma_B,
            "gate": gate,
            "gamma": gamma,
            "beta": beta,
            "aux_dist": aux_dist,
            "aux_zone": aux_zone,
        }
