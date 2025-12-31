"""
Fourier Feature + Quantile Regression Transformer

1. Fourier Features (Tancik et al., 2020)
   - Spectral Bias 해결: 고주파 공간 정보 학습 촉진
   - γ(v) = [cos(2πBv), sin(2πBv)]

2. Quantile Regression (Pinball Loss)
   - Non-Gaussianity 해결: 비대칭 오차 분포 수용
   - 분위수 0.1, 0.5, 0.9 예측
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
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

DATA_DIR = Path('/workspace/SoccerPredict/open_track1')
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


class FourierFeatureLayer(nn.Module):
    """
    Random Fourier Features for Positional Encoding
    γ(v) = [cos(2πBv), sin(2πBv)]

    고주파 공간 정보를 신경망이 빠르게 학습하도록 지원
    """
    def __init__(self, input_dim, mapping_size=64, scale=10.0):
        super().__init__()
        # B: Gaussian 분포에서 샘플링된 주파수 행렬
        B = torch.randn(input_dim, mapping_size) * scale
        self.register_buffer('B', B)
        self.output_dim = mapping_size * 2  # cos + sin

    def forward(self, x):
        # x: [batch, seq, input_dim] 또는 [batch, input_dim]
        # 2πBx
        x_proj = 2 * np.pi * torch.matmul(x, self.B)
        # [cos(2πBx), sin(2πBx)]
        return torch.cat([torch.cos(x_proj), torch.sin(x_proj)], dim=-1)


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


class FourierQuantileTransformer(nn.Module):
    """
    Selective Fourier Mapping + Quantile Regression Transformer

    구조:
    - Coordinate Stream: (start_x, start_y, end_x, end_y) → Fourier Mapping
    - Context Stream: 나머지 피처 → Standard Processing
    - Residual: 원본 좌표도 포함
    - Final: [γ(coords), coords, context]
    """
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=2, dropout=0.2,
                 fourier_mapping_size=32, fourier_scale=5.0):
        super().__init__()

        # 좌표 피처 인덱스 (start_x=0, start_y=1, end_x=14, end_y=15)
        self.coord_indices = [0, 1, 14, 15]  # base_features 14개 + masked_features 시작
        self.n_coords = len(self.coord_indices)

        # Fourier Feature Layer for all coordinate features
        self.fourier_layer = FourierFeatureLayer(self.n_coords, fourier_mapping_size, fourier_scale)

        # 전체 입력 크기: 원본 피처 + Fourier 피처
        # [γ(coords), original_features] 형태로 결합
        total_input_size = input_size + self.fourier_layer.output_dim

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.input_proj = nn.Linear(total_input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, K + 1, d_model))

        self.layers = nn.ModuleList([
            GatedTransformerBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # Quantile Head: 3개 분위수 (0.1, 0.5, 0.9) × 2차원 (x, y)
        self.quantiles = [0.1, 0.5, 0.9]
        self.n_quantiles = len(self.quantiles)

        self.quantile_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2 * self.n_quantiles)  # (x, y) × 3 quantiles
        )

        # Variance Head (불확실성 추정용)
        self.var_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, x, return_all=False):
        batch_size, seq_len, n_features = x.shape

        # Coordinate Stream: start_x, start_y, end_x, end_y 추출
        coord_features = x[:, :, self.coord_indices]  # [batch, seq, 4]

        # Fourier Features 계산: γ(coords)
        fourier_features = self.fourier_layer(coord_features)  # [batch, seq, fourier_dim]

        # Selective Fourier Mapping:
        # [γ(coords), original_features] - Residual connection으로 원본 좌표 유지
        x = torch.cat([fourier_features, x], dim=-1)

        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]

        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)

        cls_out = x[:, 0, :]

        # Quantile predictions: [batch, 6] → reshape to [batch, 3, 2]
        quantile_pred = self.quantile_head(cls_out)
        quantile_pred = quantile_pred.view(batch_size, self.n_quantiles, 2)  # [batch, 3, 2]

        # Variance prediction
        log_var = self.var_head(cls_out)
        sigma = torch.exp(0.5 * log_var)

        # Median (0.5 quantile) as main prediction
        median_pred = quantile_pred[:, 1, :]  # [batch, 2]

        if return_all:
            return quantile_pred, median_pred, log_var, sigma
        return median_pred, quantile_pred


def pinball_loss(y_true, y_pred, tau):
    """
    Pinball Loss for Quantile Regression
    ρ_τ(y - ŷ) = max(τ(y - ŷ), (τ-1)(y - ŷ))

    Args:
        y_true: [batch, 2]
        y_pred: [batch, 2]
        tau: quantile (0 < tau < 1)
    """
    diff = y_true - y_pred
    loss = torch.max(tau * diff, (tau - 1) * diff)
    return loss.mean()


def quantile_loss(y_true, quantile_preds, quantiles):
    """
    Combined Pinball Loss for multiple quantiles

    Args:
        y_true: [batch, 2]
        quantile_preds: [batch, n_quantiles, 2]
        quantiles: list of quantile values
    """
    total_loss = 0
    for i, tau in enumerate(quantiles):
        q_pred = quantile_preds[:, i, :]
        total_loss += pinball_loss(y_true, q_pred, tau)
    return total_loss / len(quantiles)


def prepare_sequence_data(df, k=K):
    base_features = ['start_x', 'start_y', 'dt', 'ep_idx_norm', 'x_zone', 'lane',
                     'dist_to_goal', 'angle_to_goal', 'type_id', 'res_id', 'is_home',
                     'pressure_x_weight', 'is_zone14', 'angle_visible']
    masked_features = ['end_x', 'end_y', 'dx', 'dy', 'dist', 'speed',
                       'action_angle', 'action_progress', 'action_dist', 'action_lateral']

    n_samples = len(df)
    n_base = len(base_features)
    n_features = n_base + len(masked_features)
    X_seq = np.zeros((n_samples, k, n_features))

    feature_indices = {
        'start_x': [0], 'start_y': [1], 'x_zone': [4], 'lane': [5],
        'dist_to_goal': [6], 'angle_to_goal': [7],
        'end_x': [n_base], 'end_y': [n_base + 1],
        'dx': [n_base + 2], 'dy': [n_base + 3],
        'action_angle': [n_base + 6],
        'action_lateral': [n_base + 9],
    }

    for t in range(k):
        for j, feat in enumerate(base_features):
            col = f'{feat}_{t}'
            if col in df.columns:
                X_seq[:, t, j] = df[col].fillna(0).values
        for j, feat in enumerate(masked_features):
            col = f'{feat}_{t}'
            if col in df.columns and t < k - 1:
                X_seq[:, t, n_base + j] = df[col].fillna(0).values

    return X_seq, feature_indices


def augment_data(X, y_coords, feature_indices):
    """Y-flip 증강"""
    X_all = [X.copy()]
    y_coords_all = [y_coords.copy()]

    X_yflip = X.copy()
    for idx in feature_indices.get('start_y', []): X_yflip[:, :, idx] = FIELD_Y - X_yflip[:, :, idx]
    for idx in feature_indices.get('end_y', []): X_yflip[:, :, idx] = FIELD_Y - X_yflip[:, :, idx]
    for idx in feature_indices.get('dy', []): X_yflip[:, :, idx] = -X_yflip[:, :, idx]
    for idx in feature_indices.get('lane', []):
        lane = X_yflip[:, :, idx].copy()
        X_yflip[:, :, idx] = np.where(lane == 0, 2, np.where(lane == 2, 0, lane))
    for idx in feature_indices.get('angle_to_goal', []): X_yflip[:, :, idx] = -X_yflip[:, :, idx]
    for idx in feature_indices.get('action_angle', []): X_yflip[:, :, idx] = -X_yflip[:, :, idx]
    for idx in feature_indices.get('action_lateral', []): X_yflip[:, :, idx] = -X_yflip[:, :, idx]

    y_coords_yflip = y_coords.copy()
    y_coords_yflip[:, 1] = FIELD_Y - y_coords_yflip[:, 1]

    X_all.append(X_yflip)
    y_coords_all.append(y_coords_yflip)

    return np.concatenate(X_all), np.concatenate(y_coords_all)


def create_boundary_labels(end_x, end_y):
    labels = np.zeros(len(end_x), dtype=int)
    labels[(end_x > FIELD_X - 5) | (end_x < 5)] = 3
    labels[(labels == 0) & (end_y > FIELD_Y - 5)] = 1
    labels[(labels == 0) & (end_y < 5)] = 2
    return labels


def main():
    log("=" * 70)
    log("Fourier Feature + Quantile Regression Transformer")
    log("  1. Fourier Features: Spectral Bias 해결")
    log("  2. Quantile Regression: Non-Gaussianity 해결")
    log("=" * 70)

    # 데이터 로드
    log("\n[1] 데이터 로드...")
    df_all = pd.read_csv(DATA_DIR / 'train_features_v2.csv')
    X_seq, feature_indices = prepare_sequence_data(df_all, K)
    y_coords = df_all[['target_end_x', 'target_end_y']].values
    y_labels = create_boundary_labels(df_all['target_end_x'].values, df_all['target_end_y'].values)

    train_idx, val_idx = train_test_split(range(len(df_all)), test_size=0.2, random_state=42)
    X_train_orig, X_val = X_seq[train_idx], X_seq[val_idx]
    y_coords_train_orig, y_coords_val = y_coords[train_idx], y_coords[val_idx]
    y_labels_val = y_labels[val_idx]

    log(f"  Train: {len(X_train_orig)}, Val: {len(X_val)}")

    # 증강
    log("\n[2] 데이터 증강 (Y-flip, 2배)...")
    X_train, y_coords_train = augment_data(X_train_orig, y_coords_train_orig, feature_indices)
    log(f"  Train (augmented): {len(X_train)}")

    # 정규화
    n_samples, n_steps, n_features = X_train.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_samples, n_steps, n_features)
    X_val = scaler.transform(X_val.reshape(-1, n_features)).reshape(len(X_val), n_steps, n_features)

    coord_mean = np.array([FIELD_X / 2, FIELD_Y / 2])
    coord_std = np.array([FIELD_X / 2, FIELD_Y / 2])
    y_coords_train_norm = (y_coords_train - coord_mean) / coord_std
    y_coords_val_norm = (y_coords_val - coord_mean) / coord_std

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"  Device: {device}")

    # 학습
    log("\n[3] Fourier Quantile Transformer 학습...")

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_coords_train_norm).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_coords_val_norm).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = FourierQuantileTransformer(
        input_size=n_features,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.3,
        fourier_mapping_size=32,
        fourier_scale=5.0  # 낮춤 (10 → 5)
    ).to(device)

    log(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    log(f"  Fourier mapping: 4 (start_x,y + end_x,y) → {model.fourier_layer.output_dim}")

    # Loss weights
    w_quantile = 1.0
    w_median_mse = 5.0
    log(f"  Loss: {w_quantile}*Pinball + {w_median_mse}*MSE(median)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_dist = float('inf')
    best_state = None
    patience, patience_counter = 25, 0

    for epoch in range(150):
        model.train()
        total_pinball, total_mse = 0, 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            quantile_pred, median_pred, log_var, sigma = model(X_batch, return_all=True)

            # Pinball Loss (Quantile Regression)
            pinball = quantile_loss(y_batch, quantile_pred, model.quantiles)

            # MSE Loss for median
            mse = F.mse_loss(median_pred, y_batch)

            loss = w_quantile * pinball + w_median_mse * mse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_pinball += pinball.item()
            total_mse += mse.item()

        model.eval()
        with torch.no_grad():
            quantile_pred, median_pred, log_var, sigma = model(X_val_t, return_all=True)

            median_orig = median_pred.cpu().numpy() * coord_std + coord_mean
            median_dist = np.sqrt(((median_orig - y_coords_val) ** 2).sum(axis=1)).mean()

            # Quantile predictions
            q_preds = quantile_pred.cpu().numpy() * coord_std + coord_mean

        scheduler.step(median_dist)

        if median_dist < best_dist:
            best_dist = median_dist
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            n_batches = len(train_loader)
            log(f"  Epoch {epoch+1:3d}: Pinball={total_pinball/n_batches:.4f}, "
                f"MSE={total_mse/n_batches:.4f}, Median={median_dist:.2f}m")

        if patience_counter >= patience:
            log(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    log(f"\n  Best Median Distance: {best_dist:.4f}m")

    # =========================================================================
    # 최종 평가
    # =========================================================================
    log("\n[4] 최종 평가...")
    model.eval()
    with torch.no_grad():
        quantile_pred, median_pred, log_var, sigma = model(X_val_t, return_all=True)

        q_preds = quantile_pred.cpu().numpy() * coord_std + coord_mean  # [N, 3, 2]
        median_np = median_pred.cpu().numpy() * coord_std + coord_mean
        sigma_np = sigma.cpu().numpy()

    # Quantile 예측값 추출
    q10 = q_preds[:, 0, :]  # 10% quantile
    q50 = q_preds[:, 1, :]  # 50% quantile (median)
    q90 = q_preds[:, 2, :]  # 90% quantile

    # Regression Performance
    median_dist = np.sqrt(((median_np - y_coords_val) ** 2).sum(axis=1)).mean()

    log("\n" + "=" * 70)
    log("[Regression Performance]")
    log("=" * 70)
    log(f"  Median (Q50) Distance: {median_dist:.4f}m")

    # Quantile Interval Width (불확실성 지표)
    interval_width_x = q90[:, 0] - q10[:, 0]
    interval_width_y = q90[:, 1] - q10[:, 1]
    log(f"\n  [80% Prediction Interval Width (Q90 - Q10)]")
    log(f"    X: mean={interval_width_x.mean():.2f}m, std={interval_width_x.std():.2f}m")
    log(f"    Y: mean={interval_width_y.mean():.2f}m, std={interval_width_y.std():.2f}m")

    # Coverage 분석 (실제값이 예측 구간 내에 있는 비율)
    in_interval_x = (y_coords_val[:, 0] >= q10[:, 0]) & (y_coords_val[:, 0] <= q90[:, 0])
    in_interval_y = (y_coords_val[:, 1] >= q10[:, 1]) & (y_coords_val[:, 1] <= q90[:, 1])
    coverage_x = in_interval_x.mean()
    coverage_y = in_interval_y.mean()
    log(f"\n  [80% Interval Coverage (목표: 0.80)]")
    log(f"    X: {coverage_x:.4f}")
    log(f"    Y: {coverage_y:.4f}")

    # Quantile Crossing 모니터링
    # 이론적으로 Q0.1 ≤ Q0.5 ≤ Q0.9 이어야 함
    crossing_x_10_50 = (q10[:, 0] > q50[:, 0]).sum()
    crossing_x_50_90 = (q50[:, 0] > q90[:, 0]).sum()
    crossing_y_10_50 = (q10[:, 1] > q50[:, 1]).sum()
    crossing_y_50_90 = (q50[:, 1] > q90[:, 1]).sum()
    total_crossings = crossing_x_10_50 + crossing_x_50_90 + crossing_y_10_50 + crossing_y_50_90
    crossing_rate = total_crossings / (len(q10) * 4)

    log(f"\n  [Quantile Crossing 모니터링 (목표: 0)]")
    log(f"    X: Q10>Q50={crossing_x_10_50}, Q50>Q90={crossing_x_50_90}")
    log(f"    Y: Q10>Q50={crossing_y_10_50}, Q50>Q90={crossing_y_50_90}")
    log(f"    Total Crossing Rate: {crossing_rate:.4f} ({total_crossings}/{len(q10)*4})")
    if total_crossings == 0:
        log(f"    ✓ Quantile 순서 유지됨!")
    else:
        log(f"    ⚠ Quantile Crossing 발생 - 모델 불안정")

    # 불확실성-오차 상관관계
    log("\n" + "=" * 70)
    log("[Uncertainty-Error Correlation]")
    log("=" * 70)
    errors = np.sqrt(((median_np - y_coords_val) ** 2).sum(axis=1))
    interval_total = np.sqrt(interval_width_x**2 + interval_width_y**2)
    correlation = np.corrcoef(errors, interval_total)[0, 1]
    log(f"  Pearson Correlation (error vs interval width): {correlation:.4f}")
    if correlation >= 0.6:
        log(f"  ✓ 목표 (≥0.6) 충족!")
    else:
        log(f"  ✗ 목표: ≥0.6")

    # 클래스별 분석
    label_names = ['In-field', 'Top-out', 'Bottom-out', 'Goal-line']
    log("\n" + "=" * 70)
    log("[클래스별 분석]")
    log("=" * 70)
    log(f"  {'Class':12s} | {'N':>5s} | {'Median':>8s} | {'Int.W.X':>8s} | {'Int.W.Y':>8s} | {'Cov.X':>6s} | {'Cov.Y':>6s}")
    log("-" * 80)

    for i, name in enumerate(label_names):
        mask = y_labels_val == i
        if mask.sum() > 0:
            med_d = np.sqrt(((median_np[mask] - y_coords_val[mask]) ** 2).sum(axis=1)).mean()
            int_x = interval_width_x[mask].mean()
            int_y = interval_width_y[mask].mean()
            cov_x = in_interval_x[mask].mean()
            cov_y = in_interval_y[mask].mean()
            log(f"  {name:12s} | {mask.sum():5d} | {med_d:8.2f} | {int_x:8.2f} | {int_y:8.2f} | {cov_x:6.2f} | {cov_y:6.2f}")

    # Interval Width 분위수별 오차
    log("\n" + "=" * 70)
    log("[Interval Width 분위수별 오차]")
    log("=" * 70)
    log(f"  {'Quantile':12s} | {'N':>5s} | {'Median Dist':>11s} | {'Avg Int.Width':>13s}")
    log("-" * 55)

    quantile_edges = [0, 0.25, 0.5, 0.75, 1.0]
    for i in range(len(quantile_edges) - 1):
        low_q = np.quantile(interval_total, quantile_edges[i])
        high_q = np.quantile(interval_total, quantile_edges[i + 1])

        if i == len(quantile_edges) - 2:
            mask = interval_total >= low_q
        else:
            mask = (interval_total >= low_q) & (interval_total < high_q)

        if mask.sum() > 0:
            med_err = np.sqrt(((median_np[mask] - y_coords_val[mask]) ** 2).sum(axis=1)).mean()
            avg_int = interval_total[mask].mean()
            label = f"[{quantile_edges[i]:.0%}-{quantile_edges[i+1]:.0%}]"
            log(f"  {label:12s} | {mask.sum():5d} | {med_err:11.2f} | {avg_int:13.2f}")

    # 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'coord_mean': coord_mean,
        'coord_std': coord_std,
        'best_dist': best_dist,
        'quantiles': model.quantiles
    }, DATA_DIR / 'transformer_fourier_quantile.pt')
    log(f"\n저장: {DATA_DIR / 'transformer_fourier_quantile.pt'}")

    log("\n" + "=" * 70)
    log("완료!")
    log("=" * 70)


if __name__ == '__main__':
    main()
