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


class HeteroscedasticTransformer(nn.Module):
    """
    Heteroscedastic Transformer with Uncertainty-Aware Residual

    Outputs:
    - Mean Head: μ(x) - 좌표 예측 (2D)
    - Variance Head: s(x) = log(σ²(x)) - 불확실성 예측 (2D, log scale)
    - Residual Head: r(x) - 잔차 예측 (2D)

    Final: y = μ + σ ⊙ r
    """
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=2, dropout=0.2):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.input_proj = nn.Linear(input_size, d_model)
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
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, x, return_all=False):
        batch_size = x.size(0)

        x = self.input_proj(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :x.size(1), :]

        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)

        cls_out = x[:, 0, :]

        # Multi-head outputs
        mu = self.mean_head(cls_out)           # μ(x): [batch, 2]
        log_var = self.var_head(cls_out)       # s(x) = log(σ²): [batch, 2]
        residual = self.residual_head(cls_out) # r(x): [batch, 2]

        # σ = exp(s/2) = exp(log(σ²)/2)
        sigma = torch.exp(0.5 * log_var)

        # Final prediction: y = μ + σ ⊙ r
        y_final = mu + sigma * residual

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
        'dx': [n_base + 2], 'dy': [n_base + 3]
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

    y_coords_yflip = y_coords.copy()
    y_coords_yflip[:, 1] = FIELD_Y - y_coords_yflip[:, 1]

    X_all.append(X_yflip)
    y_coords_all.append(y_coords_yflip)

    return np.concatenate(X_all), np.concatenate(y_coords_all)


def create_boundary_labels(end_x, end_y):
    """경계 라벨 (분석용)"""
    labels = np.zeros(len(end_x), dtype=int)
    labels[(end_x > FIELD_X - 5) | (end_x < 5)] = 3
    labels[(labels == 0) & (end_y > FIELD_Y - 5)] = 1
    labels[(labels == 0) & (end_y < 5)] = 2
    return labels


def main():
    log("=" * 70)
    log("Heteroscedastic Transformer: Aleatoric Uncertainty 기반 회귀")
    log("Based on Kendall & Gal (2017)")
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
    log("\n[3] Heteroscedastic Transformer 학습...")

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_coords_train_norm).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_val_t = torch.FloatTensor(y_coords_val_norm).to(device)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = HeteroscedasticTransformer(
        input_size=n_features,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.3
    ).to(device)

    log(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

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

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()

            mu, log_var, sigma, residual, y_final = model(X_batch, return_all=True)

            # NLL Loss: μ로 분포 학습, σ로 불확실성 인지
            nll_base = gaussian_nll_loss(y_batch, mu, log_var)

            # MSE Loss: 최종 예측 정밀도 (μ + σ⊙r)
            mse_final = F.mse_loss(y_final, y_batch)

            loss = w_nll * nll_base + w_final * mse_final

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_nll += nll_base.item()
            total_mse += mse_final.item()

        model.eval()
        with torch.no_grad():
            mu, log_var, sigma, residual, y_final = model(X_val_t, return_all=True)

            # Base (μ) distance
            mu_orig = mu.cpu().numpy() * coord_std + coord_mean
            mu_dist = np.sqrt(((mu_orig - y_coords_val) ** 2).sum(axis=1)).mean()

            # Final (μ + σ⊙r) distance
            final_orig = y_final.cpu().numpy() * coord_std + coord_mean
            final_dist = np.sqrt(((final_orig - y_coords_val) ** 2).sum(axis=1)).mean()

        scheduler.step(final_dist)

        if final_dist < best_dist:
            best_dist = final_dist
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            n_batches = len(train_loader)
            avg_sigma = sigma.mean().item()
            log(f"  Epoch {epoch+1:3d}: NLL={total_nll/n_batches:.4f}, "
                f"μ={mu_dist:.2f}m, Final={final_dist:.2f}m, σ_avg={avg_sigma:.3f}")

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
        mu, log_var, sigma, residual, y_final = model(X_val_t, return_all=True)

        mu_np = mu.cpu().numpy() * coord_std + coord_mean
        sigma_np = sigma.cpu().numpy()
        residual_np = residual.cpu().numpy()
        final_np = y_final.cpu().numpy() * coord_std + coord_mean

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
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'coord_mean': coord_mean,
        'coord_std': coord_std,
        'best_dist': best_dist
    }, DATA_DIR / 'transformer_heteroscedastic.pt')
    log(f"\n저장: {DATA_DIR / 'transformer_heteroscedastic.pt'}")

    log("\n" + "=" * 70)
    log("완료!")
    log("=" * 70)


if __name__ == '__main__':
    main()
