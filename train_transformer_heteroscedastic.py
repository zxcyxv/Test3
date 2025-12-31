"""
Heteroscedastic Transformer: Aleatoric Uncertainty 기반 회귀
Based on Kendall & Gal (2017) "What Uncertainties Do We Need in Bayesian Deep Learning?"

핵심 아이디어:
- 분류기 기반 게이팅 폐기
- 모델이 직접 예측 불확실성 σ²(x)를 학습
- 불확실성이 높은 영역에서 Residual Head가 보정

Loss 구조 (Gradient 정렬):
- NLL_base: μ로 대략적인 분포 학습, σ로 불확실성 인지
- MSE_final: μ + σ⊙r 로 정밀 보정

Final prediction:
y_final = y_base + σ_base ⊙ y_residual
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

    def forward(self, x, src_key_padding_mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed, key_padding_mask=src_key_padding_mask
        )
        gate = self.gate(normed)
        self.last_gate_mean = gate.mean().item()
        self.last_gate_low = (gate < 0.05).float().mean().item()
        self.last_gate_high = (gate > 0.95).float().mean().item()
        attn_out = gate * attn_out
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class FourierFeatureLayer(nn.Module):
    """
    Random Fourier Features for Positional Encoding
    γ(v) = [cos(2πBv), sin(2πBv)]

    Spectral Bias 해결: 고주파 공간 정보 학습 촉진

    Coordinates are assumed normalized already.
    """
    def __init__(self, input_dim, mapping_size=32, scale=5.0):
        super().__init__()
        # 랜덤 주파수 행렬 (deterministic seed)
        torch.manual_seed(42)
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer('B', B)
        self.output_dim = mapping_size * 2

    def forward(self, x):
        # Fourier projection
        x_proj = 2 * np.pi * torch.matmul(x, self.B)
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


class HeteroscedasticTransformer(nn.Module):
    """
    Fourier + Heteroscedastic Transformer

    1. Selective Fourier Mapping: 좌표 피처에 고주파 인코딩
    2. Heteroscedastic Regression: 불확실성 기반 잔차 보정

    Final: y = μ + σ ⊙ r
    """
    def __init__(self, num_cont_features, cat_cardinalities,
                 d_model=128, nhead=4, num_layers=2, dropout=0.2,
                 fourier_mapping_size=32, fourier_scale=5.0):
        super().__init__()

        self.coord_dim = 4

        # Categorical embeddings
        self.embedding_dim = 8
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, self.embedding_dim, padding_idx=0)
            for cardinality in cat_cardinalities
        ])
        total_emb_dim = len(cat_cardinalities) * self.embedding_dim

        # Fourier Feature Layer
        self.fourier_layer = FourierFeatureLayer(self.coord_dim, fourier_mapping_size, fourier_scale)

        # 전체 입력: [γ(coords), original_features]
        total_input_size = (
            self.coord_dim + self.fourier_layer.output_dim +
            num_cont_features + total_emb_dim
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.input_proj = nn.Linear(total_input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, K + 1, d_model))

        self.layers = nn.ModuleList([
            GatedTransformerBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # Mean Head: μ(x) - 좌표 예측
        self.mean_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)
        )

        # Variance Head: s(x) = log(σ²(x)) - 불확실성 예측
        self.var_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)
        )

        # Residual Head: r(x) - 잔차 예측 (불확실성 높은 영역 전담)
        # Capacity 증가: 더 깊고 넓은 네트워크
        self.residual_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, x_coords, x_cont, x_cat, padding_mask, return_all=False):
        batch_size, seq_len, _ = x_coords.shape

        # Fourier Mapping: γ(coords)
        fourier_features = self.fourier_layer(x_coords)  # [batch, seq, fourier_dim]

        # Categorical embeddings
        emb_features = [
            emb(x_cat[:, :, i])
            for i, emb in enumerate(self.embeddings)
        ]
        x_emb = torch.cat(emb_features, dim=-1) if emb_features else x_coords.new_zeros(batch_size, seq_len, 0)

        # Selective Fourier Mapping: [coords, γ(coords), cont, emb]
        x = torch.cat([x_coords, fourier_features, x_cont, x_emb], dim=-1)

        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]

        cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
        attn_mask = torch.cat([cls_mask, padding_mask], dim=1)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attn_mask)
        x = self.final_norm(x)

        cls_out = x[:, 0, :]
        token_out = x[:, 1:, :]
        valid_tokens = ~padding_mask
        last_idx = valid_tokens.long().sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(batch_size, device=x.device)
        last_out = token_out[batch_idx, last_idx]
        self.last_cls_norm = cls_out.norm(dim=1).mean().item()
        self.last_token_norm = last_out.norm(dim=1).mean().item()

        # Multi-head outputs
        mu = self.mean_head(cls_out)           # μ(x): [batch, 2]
        log_var = self.var_head(cls_out)       # s(x) = log(σ²): [batch, 2]
        residual = self.residual_head(last_out) # r(x): [batch, 2]

        # σ = exp(s/2) = exp(log(σ²)/2)
        sigma = torch.exp(0.5 * log_var)

        # Final prediction: y = μ + σ ⊙ r (stop σ gradient to isolate residual learning)
        y_final = mu + sigma.detach() * residual

        if return_all:
            return mu, log_var, sigma, residual, y_final
        return y_final, log_var


def gaussian_nll_loss(y_true, mu, log_var):
    """
    Gaussian Negative Log-Likelihood Loss
    L = exp(-s) * ||y - μ||² + s
    where s = log(σ²)
    """
    precision = torch.exp(-log_var)  # 1/σ²
    mse = (y_true - mu) ** 2
    loss = precision * mse + log_var
    return loss.mean()


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = x.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    return ranks


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
                scale = FIELD_X if feat.endswith('x') else FIELD_Y
                vals = df.loc[has_data, col].values / scale
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

    return (
        coords_seq,
        cont_seq,
        angle_seq,
        cat_seq,
        valid_mask_seq,
        cont_valid_mask,
        cont_idx,
        angle_idx,
        cat_idx,
    )


def augment_data(coords, cont, angles, cat, padding_mask, cont_valid_mask,
                 y_coords, cont_idx, angle_idx, cat_idx):
    """Y-flip 증강"""
    coords_all = [coords.copy()]
    cont_all = [cont.copy()]
    angles_all = [angles.copy()]
    cat_all = [cat.copy()]
    mask_all = [padding_mask.copy()]
    cont_valid_all = [cont_valid_mask.copy()]
    y_coords_all = [y_coords.copy()]

    coords_yflip = coords.copy()
    y_max_norm = 1.0
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
        lane_swapped = np.where(lane == 1, 3, np.where(lane == 3, 1, lane))
        cat_yflip[:, :, cat_idx['lane']] = np.where(valid_mask, lane_swapped, lane)

    y_coords_yflip = y_coords.copy()
    y_coords_yflip[:, 1] = FIELD_Y - y_coords_yflip[:, 1]

    coords_all.append(coords_yflip)
    cont_all.append(cont_yflip)
    angles_all.append(angles_yflip)
    cat_all.append(cat_yflip)
    mask_all.append(padding_mask.copy())
    cont_valid_all.append(cont_valid_mask.copy())
    y_coords_all.append(y_coords_yflip)

    return (np.concatenate(coords_all), np.concatenate(cont_all),
            np.concatenate(angles_all), np.concatenate(cat_all),
            np.concatenate(mask_all), np.concatenate(cont_valid_all),
            np.concatenate(y_coords_all))


def create_boundary_labels(end_x, end_y):
    """경계 라벨 (분석용)"""
    labels = np.zeros(len(end_x), dtype=int)
    labels[(end_x > FIELD_X - 5) | (end_x < 5)] = 3
    labels[(labels == 0) & (end_y > FIELD_Y - 5)] = 1
    labels[(labels == 0) & (end_y < 5)] = 2
    return labels


def main():
    log("=" * 70)
    log("Fourier + Heteroscedastic Transformer")
    log("  1. Selective Fourier Mapping: Spectral Bias 해결")
    log("  2. Heteroscedastic Regression: Aleatoric Uncertainty")
    log("=" * 70)

    # 데이터 로드
    log("\n[1] 데이터 로드...")
    df_all = pd.read_csv(DATA_DIR / 'train_features_v2.csv')
    (coords_seq, cont_seq, angles_seq, cat_seq, valid_mask_seq,
     cont_valid_mask_seq, cont_idx, angle_idx, cat_idx) = prepare_sequence_data(df_all, K)
    y_coords = df_all[['target_end_x', 'target_end_y']].values
    y_labels = create_boundary_labels(df_all['target_end_x'].values, df_all['target_end_y'].values)

    train_idx, val_idx = train_test_split(range(len(df_all)), test_size=0.2, random_state=42)
    coords_train_orig, coords_val = coords_seq[train_idx], coords_seq[val_idx]
    cont_train_orig, cont_val = cont_seq[train_idx], cont_seq[val_idx]
    angles_train_orig, angles_val = angles_seq[train_idx], angles_seq[val_idx]
    cat_train_orig, cat_val = cat_seq[train_idx], cat_seq[val_idx]
    valid_mask_train_orig, valid_mask_val = valid_mask_seq[train_idx], valid_mask_seq[val_idx]
    cont_valid_mask_train_orig, cont_valid_mask_val = cont_valid_mask_seq[train_idx], cont_valid_mask_seq[val_idx]
    y_coords_train_orig, y_coords_val = y_coords[train_idx], y_coords[val_idx]
    y_labels_val = y_labels[val_idx]

    log(f"  Train: {len(coords_train_orig)}, Val: {len(coords_val)}")

    # 증강
    log("\n[2] 데이터 증강 (Y-flip, 2배)...")
    padding_mask_train_orig = ~valid_mask_train_orig
    padding_mask_val = ~valid_mask_val

    coords_train, cont_train, angles_train, cat_train, padding_mask_train, cont_valid_mask_train, y_coords_train = augment_data(
        coords_train_orig, cont_train_orig, angles_train_orig, cat_train_orig,
        padding_mask_train_orig, cont_valid_mask_train_orig,
        y_coords_train_orig, cont_idx, angle_idx, cat_idx
    )
    log(f"  Train (augmented): {len(coords_train)}")

    # 정규화
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

    coord_scale = np.array([FIELD_X, FIELD_Y])
    y_coords_train_norm = y_coords_train / coord_scale
    y_coords_val_norm = y_coords_val / coord_scale

    cat_cardinalities = tuple(
        int(cat_train_orig[:, :, i].max()) + 1 for i in range(cat_train_orig.shape[2])
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"  Device: {device}")

    # 학습
    log("\n[3] Heteroscedastic Transformer 학습...")

    coords_train_t = torch.FloatTensor(coords_train).to(device)
    cont_train_t = torch.FloatTensor(cont_train).to(device)
    cat_train_t = torch.LongTensor(cat_train).to(device)
    padding_mask_train_t = torch.BoolTensor(padding_mask_train).to(device)
    y_train_t = torch.FloatTensor(y_coords_train_norm).to(device)
    coords_val_t = torch.FloatTensor(coords_val).to(device)
    cont_val_t = torch.FloatTensor(cont_val).to(device)
    cat_val_t = torch.LongTensor(cat_val).to(device)
    padding_mask_val_t = torch.BoolTensor(padding_mask_val).to(device)
    y_val_t = torch.FloatTensor(y_coords_val_norm).to(device)

    train_dataset = TensorDataset(
        coords_train_t, cont_train_t, cat_train_t, padding_mask_train_t, y_train_t
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    full_cont_dim = cont_train.shape[2]

    model = HeteroscedasticTransformer(
        num_cont_features=full_cont_dim,
        cat_cardinalities=cat_cardinalities,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.3,
        fourier_mapping_size=64,
        fourier_scale=3.0
    ).to(device)

    log(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    log(f"  Fourier mapping: 4 coords → {model.fourier_layer.output_dim}")

    # Loss 구조 (Gradient 정렬)
    # - NLL_base: μ로 대략적인 분포 학습 + σ로 불확실성 인지
    # - MSE_final: μ + σ⊙r 로 정밀 보정
    w_nll = 1.0      # NLL loss for base (μ, σ)
    w_final = 5.0    # MSE loss for final (μ + σ⊙r)
    log(f"  Loss: {w_nll}*NLL(μ,σ) + {w_final}*MSE(μ+σ⊙r)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_dist = float('inf')
    best_state = None
    patience, patience_counter = 25, 0

    for epoch in range(150):
        model.train()
        total_nll, total_mse = 0, 0

        for coords_batch, cont_batch, cat_batch, mask_batch, y_batch in train_loader:
            optimizer.zero_grad()

            mu, log_var, sigma, residual, y_final = model(
                coords_batch, cont_batch, cat_batch, mask_batch, return_all=True
            )
            if not torch.isfinite(mu).all() or not torch.isfinite(log_var).all():
                log("  [ERR] NaN/Inf in mu/log_var. Stopping.")
                return
            if not torch.isfinite(sigma).all() or not torch.isfinite(residual).all():
                log("  [ERR] NaN/Inf in sigma/residual. Stopping.")
                return
            if not torch.isfinite(y_final).all():
                log("  [ERR] NaN/Inf in y_final. Stopping.")
                return

            # NLL Loss: μ로 분포 학습, σ로 불확실성 인지
            nll_base = gaussian_nll_loss(y_batch, mu, log_var)

            # MSE Loss: 최종 예측 정밀도 (μ + σ⊙r)
            mse_final = F.mse_loss(y_final, y_batch)

            loss = w_nll * nll_base + w_final * mse_final
            if not torch.isfinite(loss).all():
                log("  [ERR] NaN/Inf in loss. Stopping.")
                return

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_nll += nll_base.item()
            total_mse += mse_final.item()

        model.eval()
        with torch.no_grad():
            mu, log_var, sigma, residual, y_final = model(
                coords_val_t, cont_val_t, cat_val_t, padding_mask_val_t, return_all=True
            )

            # Base (μ) distance
            mu_orig = mu.cpu().numpy() * coord_scale
            mu_dist = np.sqrt(((mu_orig - y_coords_val) ** 2).sum(axis=1)).mean()

            # Final (μ + σ⊙r) distance
            final_orig = y_final.cpu().numpy() * coord_scale
            final_dist = np.sqrt(((final_orig - y_coords_val) ** 2).sum(axis=1)).mean()

            # Monitoring stats
            seq_len = (~padding_mask_val_t).sum(dim=1).cpu().numpy()
            seq_mean = float(seq_len.mean())
            seq_min = int(seq_len.min())
            seq_max = int(seq_len.max())
            seq_zero = int((seq_len == 0).sum())

            mu_np = mu.cpu().numpy()
            sigma_np = sigma.cpu().numpy()
            residual_np = residual.cpu().numpy()
            err = mu_orig - y_coords_val
            err_norm = np.sqrt((err ** 2).sum(axis=1))
            sigma_norm = np.sqrt((sigma_np ** 2).sum(axis=1))
            res_norm = np.sqrt((residual_np ** 2).sum(axis=1))
            sigma_res_norm = np.sqrt(((sigma_np * residual_np) ** 2).sum(axis=1))
            c_vec = sigma_np * residual_np
            e_norm = np.linalg.norm(err, axis=1)
            c_norm = np.linalg.norm(c_vec, axis=1)
            denom = (e_norm * c_norm) + 1e-8
            align = float(np.mean((err * c_vec).sum(axis=1) / denom))

            z = (y_coords_val / coord_scale - mu_np) / (sigma_np + 1e-6)
            z_mean = float(np.mean(z))
            z_var = float(np.var(z))
            cover_1 = float(np.mean(np.abs(y_coords_val / coord_scale - mu_np) <= sigma_np))
            cover_2 = float(np.mean(np.abs(y_coords_val / coord_scale - mu_np) <= 2 * sigma_np))

            corr_sigma = float(np.corrcoef(err_norm, sigma_norm)[0, 1])
            corr_sigma_res = float(np.corrcoef(err_norm, sigma_res_norm)[0, 1])
            spearman_sigma = float(np.corrcoef(_rankdata(err_norm), _rankdata(sigma_norm))[0, 1])

            r_target = (y_coords_val / coord_scale - mu_np) / (sigma_np + 1e-6)
            mse_r = float(np.mean((residual_np - r_target) ** 2))
            corr_r = float(np.corrcoef(np.linalg.norm(r_target, axis=1), res_norm)[0, 1])

            p75 = np.quantile(sigma_norm, 0.75)
            top_mask = sigma_norm >= p75
            if top_mask.any():
                top_align = float(np.mean((err[top_mask] * c_vec[top_mask]).sum(axis=1) /
                                          ((e_norm[top_mask] * c_norm[top_mask]) + 1e-8)))
                top_delta = float(
                    np.sqrt(((mu_orig[top_mask] - y_coords_val[top_mask]) ** 2).sum(axis=1)).mean()
                    - np.sqrt(((final_orig[top_mask] - y_coords_val[top_mask]) ** 2).sum(axis=1)).mean()
                )
                top_sigma_res = float(sigma_res_norm[top_mask].mean())
                top_mse_r = float(np.mean((residual_np[top_mask] - r_target[top_mask]) ** 2))
            else:
                top_align = 0.0
                top_delta = 0.0
                top_sigma_res = 0.0
                top_mse_r = 0.0

        scheduler.step(final_dist)

        if final_dist < best_dist:
            best_dist = final_dist
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            n_batches = len(train_loader)
            avg_sigma = sigma.mean().item()
            log(f"  Epoch {epoch+1:3d}: NLL={total_nll/n_batches:.4f}, "
                f"μ={mu_dist:.2f}m, Final={final_dist:.2f}m, σ_avg={avg_sigma:.3f}")
            log(f"    Δ(μ-Final)={mu_dist - final_dist:+.3f}m, seq_len={seq_mean:.2f} [{seq_min},{seq_max}], zeros={seq_zero}")
            log(f"    σ_mean={sigma_norm.mean():.4f}, σ_p95={np.percentile(sigma_norm, 95):.4f}, log_var_mean={log_var.mean().item():.4f}")
            log(f"    corr(|e|,||σ||)={corr_sigma:.3f} (spearman {spearman_sigma:.3f}), z_mean={z_mean:.3f}, z_var={z_var:.3f}")
            log(f"    coverage P(|e|<=σ)={cover_1:.3f}, P(|e|<=2σ)={cover_2:.3f}")
            log(f"    ||res||_mean={res_norm.mean():.4f}, ||σ⊙res||_mean={sigma_res_norm.mean():.4f}, corr(|e|,||σ⊙res||)={corr_sigma_res:.3f}")
            log(f"    align(cos e,c)={align:.3f}, mse_r={mse_r:.4f}, corr(||r_t||,||r||)={corr_r:.3f}")
            log(f"    topσ25 Δ={top_delta:+.3f} align={top_align:.3f} ||σ⊙res||={top_sigma_res:.4f} mse_r={top_mse_r:.4f}")
            gate_stats = [
                (layer.last_gate_mean, layer.last_gate_low, layer.last_gate_high)
                for layer in model.layers
            ]
            gate_mean = float(np.mean([g[0] for g in gate_stats]))
            gate_low = float(np.mean([g[1] for g in gate_stats]))
            gate_high = float(np.mean([g[2] for g in gate_stats]))
            log(f"    gate_mean={gate_mean:.3f}, gate_low={gate_low:.3f}, gate_high={gate_high:.3f}")
            log(f"    ||cls||={model.last_cls_norm:.3f}, ||last||={model.last_token_norm:.3f}")
            w = model.input_proj.weight.detach().cpu().numpy()
            coord_dim = model.coord_dim
            fourier_dim = model.fourier_layer.output_dim
            cont_dim = cont_train.shape[2]
            emb_dim = len(model.embeddings) * model.embedding_dim
            idx = 0
            w_coord = np.linalg.norm(w[:, idx:idx + coord_dim]); idx += coord_dim
            w_fourier = np.linalg.norm(w[:, idx:idx + fourier_dim]); idx += fourier_dim
            w_cont = np.linalg.norm(w[:, idx:idx + cont_dim]); idx += cont_dim
            w_emb = np.linalg.norm(w[:, idx:idx + emb_dim])
            log(f"    proj_norms coord={w_coord:.3f} fourier={w_fourier:.3f} cont={w_cont:.3f} emb={w_emb:.3f}")

        if patience_counter >= patience:
            log(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    log(f"\n  Best Final Distance: {best_dist:.4f}m")

    # =========================================================================
    # 최종 평가
    # =========================================================================
    log("\n[4] 최종 평가...")
    model.eval()
    with torch.no_grad():
        mu, log_var, sigma, residual, y_final = model(
            coords_val_t, cont_val_t, cat_val_t, padding_mask_val_t, return_all=True
        )

        mu_np = mu.cpu().numpy() * coord_scale
        sigma_np = sigma.cpu().numpy()
        residual_np = residual.cpu().numpy()
        final_np = y_final.cpu().numpy() * coord_scale

    # Regression Performance
    mu_dist = np.sqrt(((mu_np - y_coords_val) ** 2).sum(axis=1)).mean()
    final_dist = np.sqrt(((final_np - y_coords_val) ** 2).sum(axis=1)).mean()

    log("\n" + "=" * 70)
    log("[Regression Performance]")
    log("=" * 70)
    log(f"  Base (μ) Distance:     {mu_dist:.4f}m")
    log(f"  Final (μ+σ⊙r) Distance: {final_dist:.4f}m")
    log(f"  Improvement:           {mu_dist - final_dist:.4f}m")

    # =========================================================================
    # Winning Signal #1: Pearson Correlation (error vs σ)
    # =========================================================================
    log("\n" + "=" * 70)
    log("[Winning Signal #1] 불확실성-오차 상관관계")
    log("=" * 70)
    errors = np.sqrt(((mu_np - y_coords_val) ** 2).sum(axis=1))
    sigma_total = np.sqrt((sigma_np ** 2).sum(axis=1))
    correlation = np.corrcoef(errors, sigma_total)[0, 1]
    log(f"  Pearson Correlation (error vs σ): {correlation:.4f}")
    if correlation >= 0.6:
        log(f"  ✓ 우승권 모델 기준 (≥0.6) 충족!")
    else:
        log(f"  ✗ 목표: ≥0.6 (현재 {correlation:.4f})")

    # =========================================================================
    # Winning Signal #2: σ 분위수별 오차 감소
    # =========================================================================
    log("\n" + "=" * 70)
    log("[Winning Signal #2] σ 분위수별 오차 (상위 25%에서 Δ 최대여야 함)")
    log("=" * 70)
    log(f"  {'Quantile':12s} | {'N':>5s} | {'μ Dist':>8s} | {'Final':>8s} | {'Δ':>8s}")
    log("-" * 55)

    quantile_edges = [0, 0.25, 0.5, 0.75, 1.0]
    deltas = []
    for i in range(len(quantile_edges) - 1):
        low_q = np.quantile(sigma_total, quantile_edges[i])
        high_q = np.quantile(sigma_total, quantile_edges[i + 1])

        if i == len(quantile_edges) - 2:  # 마지막 구간
            mask = sigma_total >= low_q
        else:
            mask = (sigma_total >= low_q) & (sigma_total < high_q)

        if mask.sum() > 0:
            mu_err = np.sqrt(((mu_np[mask] - y_coords_val[mask]) ** 2).sum(axis=1)).mean()
            final_err = np.sqrt(((final_np[mask] - y_coords_val[mask]) ** 2).sum(axis=1)).mean()
            delta = mu_err - final_err
            deltas.append(delta)
            label = f"σ [{quantile_edges[i]:.0%}-{quantile_edges[i+1]:.0%}]"
            log(f"  {label:12s} | {mask.sum():5d} | {mu_err:8.2f} | {final_err:8.2f} | {delta:+8.2f}")

    if len(deltas) >= 4 and deltas[-1] == max(deltas):
        log(f"\n  ✓ 상위 25% (σ 최대)에서 Δ가 최대! ({deltas[-1]:+.2f}m)")
    else:
        log(f"\n  ✗ 상위 25%의 Δ가 최대가 아님. Residual Head capacity 증가 필요.")

    # =========================================================================
    # Uncertainty Statistics
    # =========================================================================
    log("\n" + "=" * 70)
    log("[Uncertainty (σ) Statistics]")
    log("=" * 70)
    log(f"  σ_x: mean={sigma_np[:, 0].mean():.4f}, std={sigma_np[:, 0].std():.4f}, "
        f"min={sigma_np[:, 0].min():.4f}, max={sigma_np[:, 0].max():.4f}")
    log(f"  σ_y: mean={sigma_np[:, 1].mean():.4f}, std={sigma_np[:, 1].std():.4f}, "
        f"min={sigma_np[:, 1].min():.4f}, max={sigma_np[:, 1].max():.4f}")

    # =========================================================================
    # 클래스별 분석
    # =========================================================================
    label_names = ['In-field', 'Top-out', 'Bottom-out', 'Goal-line']
    log("\n" + "=" * 70)
    log("[클래스별 분석]")
    log("=" * 70)
    log(f"  {'Class':12s} | {'N':>5s} | {'μ Dist':>8s} | {'Final':>8s} | {'Δ':>8s} | {'σ_x':>6s} | {'σ_y':>6s}")
    log("-" * 75)

    for i, name in enumerate(label_names):
        mask = y_labels_val == i
        if mask.sum() > 0:
            mu_d = np.sqrt(((mu_np[mask] - y_coords_val[mask]) ** 2).sum(axis=1)).mean()
            final_d = np.sqrt(((final_np[mask] - y_coords_val[mask]) ** 2).sum(axis=1)).mean()
            avg_sigma_x = sigma_np[mask, 0].mean()
            avg_sigma_y = sigma_np[mask, 1].mean()
            delta = mu_d - final_d
            log(f"  {name:12s} | {mask.sum():5d} | {mu_d:8.2f} | {final_d:8.2f} | {delta:+8.2f} | {avg_sigma_x:6.3f} | {avg_sigma_y:6.3f}")

    # 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'cont_mean': cont_mean,
        'cont_std': cont_std,
        'coord_scale': coord_scale,
        'cat_cardinalities': cat_cardinalities,
        'best_dist': best_dist
    }, DATA_DIR / 'transformer_heteroscedastic.pt')
    log(f"\n저장: {DATA_DIR / 'transformer_heteroscedastic.pt'}")

    log("\n" + "=" * 70)
    log("완료!")
    log("=" * 70)


if __name__ == '__main__':
    main()
