"""
Origin Router + Aux Boundary Head Model
Reconstructs the earlier high-accuracy setup:
- Fourier-based SpatialRouter
- Auxiliary 4-class boundary classification head
- MSE regression with MTL (MSE + CE + BCE)
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def log(msg: str) -> None:
    print(msg, flush=True)


DATA_DIR = Path("/workspace/SoccerPredict/open_track1")
FIELD_X, FIELD_Y = 105, 68
K = 8


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

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        attn_out = self.gate(normed) * attn_out
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class SpatialRouter(nn.Module):
    """
    Fourier-based router. Gate=1 means Boundary, Gate=0 means In-field.
    """

    def __init__(self, hidden_dim: int = 64, fourier_dim: int = 32, fourier_scale: float = 10.0):
        super().__init__()
        self.register_buffer("B", torch.randn(6, fourier_dim) * fourier_scale)
        input_dim = 6 + 2 * fourier_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, pos: torch.Tensor, temperature: float = 0.05) -> torch.Tensor:
        pos_norm = pos / pos.new_tensor([FIELD_X, FIELD_Y])
        dists = torch.stack(
            [
                pos[:, 0] / FIELD_X,
                (FIELD_X - pos[:, 0]) / FIELD_X,
                pos[:, 1] / FIELD_Y,
                (FIELD_Y - pos[:, 1]) / FIELD_Y,
            ],
            dim=-1,
        )
        x = torch.cat([pos_norm, dists], dim=-1)
        proj = 2 * np.pi * (x @ self.B)
        feat = torch.cat([x, torch.cos(proj), torch.sin(proj)], dim=-1)
        return torch.sigmoid(self.net(feat) / (temperature + 1e-6))


class AuxiliaryBoundaryHead(nn.Module):
    def __init__(self, d_model: int = 128):
        super().__init__()
        self.zone_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4),
        )

    def forward(self, cls_token: torch.Tensor) -> torch.Tensor:
        return self.zone_classifier(cls_token)


class OriginRouterModel(nn.Module):
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embedding = nn.Parameter(torch.randn(1, K + 1, d_model))
        self.layers = nn.ModuleList([GatedTransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model)

        self.reg_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2),
        )
        self.aux_head = AuxiliaryBoundaryHead(d_model)
        self.router = SpatialRouter()

    def forward(self, x: torch.Tensor, pos: torch.Tensor, temperature: float = 0.05) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]

        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        cls_out = x[:, 0, :]

        y_pred = self.reg_head(cls_out)
        zone_logits = self.aux_head(cls_out)
        gate = self.router(pos, temperature=temperature)
        return {
            "y_pred": y_pred,
            "zone_logits": zone_logits,
            "gate": gate,
        }


def create_boundary_labels(end_x, end_y):
    labels = np.zeros(len(end_x), dtype=int)
    labels[(end_x > FIELD_X - 5) | (end_x < 5)] = 3
    labels[(labels == 0) & (end_y > FIELD_Y - 5)] = 1
    labels[(labels == 0) & (end_y < 5)] = 2
    return labels


def prepare_sequence_data(df, k=K):
    base_features = [
        "start_x",
        "start_y",
        "dt",
        "ep_idx_norm",
        "x_zone",
        "lane",
        "dist_to_goal",
        "angle_to_goal",
        "type_id",
        "res_id",
        "is_home",
        "pressure_x_weight",
        "is_zone14",
        "angle_visible",
    ]
    masked_features = [
        "end_x",
        "end_y",
        "dx",
        "dy",
        "dist",
        "speed",
        "action_angle",
        "action_progress",
        "action_dist",
        "action_lateral",
    ]

    n_samples = len(df)
    n_base = len(base_features)
    n_features = n_base + len(masked_features)
    X_seq = np.zeros((n_samples, k, n_features))

    for t in range(k):
        for j, feat in enumerate(base_features):
            col = f"{feat}_{t}"
            if col in df.columns:
                X_seq[:, t, j] = df[col].fillna(0).values
        for j, feat in enumerate(masked_features):
            col = f"{feat}_{t}"
            if col in df.columns and t < k - 1:
                X_seq[:, t, n_base + j] = df[col].fillna(0).values

    return X_seq


def get_last_positions(df, k=K):
    sx = df[f"start_x_{k-1}"].fillna(0).values
    sy = df[f"start_y_{k-1}"].fillna(0).values
    return np.stack([sx, sy], axis=1)


def temperature_schedule(epoch: int, warmup=10, anneal=90):
    if epoch < warmup:
        return 2.0
    if epoch < warmup + anneal:
        progress = (epoch - warmup) / anneal
        return 2.0 * (0.05 / 2.0) ** progress
    return 0.05


def main():
    log("=" * 70)
    log("Origin Router + Aux Boundary Head")
    log("=" * 70)

    df = pd.read_csv(DATA_DIR / "train_features_v2.csv")
    X_seq = prepare_sequence_data(df, K)
    pos_all = get_last_positions(df, K)
    y_coords = df[["target_end_x", "target_end_y"]].values
    boundary_zone = create_boundary_labels(y_coords[:, 0], y_coords[:, 1])

    train_idx, val_idx = train_test_split(range(len(df)), test_size=0.2, random_state=42)
    X_train, X_val = X_seq[train_idx], X_seq[val_idx]
    pos_train, pos_val = pos_all[train_idx], pos_all[val_idx]
    y_train, y_val = y_coords[train_idx], y_coords[val_idx]
    zone_train, zone_val = boundary_zone[train_idx], boundary_zone[val_idx]

    scaler = StandardScaler()
    n_samples, n_steps, n_features = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_samples, n_steps, n_features)
    X_val = scaler.transform(X_val.reshape(-1, n_features)).reshape(len(X_val), n_steps, n_features)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"  Device: {device}")

    X_train_t = torch.FloatTensor(X_train).to(device)
    pos_train_t = torch.FloatTensor(pos_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    zone_train_t = torch.LongTensor(zone_train).to(device)

    X_val_t = torch.FloatTensor(X_val).to(device)
    pos_val_t = torch.FloatTensor(pos_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)
    zone_val_t = torch.LongTensor(zone_val).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, pos_train_t, y_train_t, zone_train_t),
        batch_size=256,
        shuffle=True,
    )

    model = OriginRouterModel(input_size=n_features, d_model=128, nhead=4, num_layers=2, dropout=0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    best = float("inf")
    for epoch in range(50):
        model.train()
        total = 0.0
        temperature = temperature_schedule(epoch)
        for X_b, p_b, y_b, z_b in train_loader:
            optimizer.zero_grad()
            out = model(X_b, p_b, temperature=temperature)
            mse = F.mse_loss(out["y_pred"], y_b)
            ce = F.cross_entropy(out["zone_logits"], z_b)
            boundary_mask = (z_b != 0).float().unsqueeze(1)
            gate_loss = F.binary_cross_entropy(out["gate"], boundary_mask)
            loss = mse + 0.5 * ce + 0.5 * gate_loss
            loss.backward()
            optimizer.step()
            total += loss.item()

        model.eval()
        with torch.no_grad():
            out = model(X_val_t, pos_val_t, temperature=0.05)
            val_mse = F.mse_loss(out["y_pred"], y_val_t).item()
            zone_pred = out["zone_logits"].argmax(dim=1)
            val_acc = (zone_pred == zone_val_t).float().mean().item()
            if val_mse < best:
                best = val_mse

        if (epoch + 1) % 10 == 0:
            log(f"  Epoch {epoch+1:3d}: train_loss={total/len(train_loader):.4f}, val_mse={val_mse:.4f}, val_acc={val_acc:.4f}")

    log(f"Best val MSE: {best:.4f}")


if __name__ == "__main__":
    main()
