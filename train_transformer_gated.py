"""
Gated Transformer: 분류 확률 기반 Gating 메커니즘 적용
- Base Regression Head: 기본 좌표 예측
- Residual Head: 잔차 예측
- Gating: y_final = y_base + (1 - P_infield) * y_residual
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
    """RMSNorm - LayerNorm보다 빠르고 효율적"""
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * (x / rms)


class GatedTransformerBlock(nn.Module):
    """Pre-Norm Transformer Block with RMSNorm + Gated Attention"""
    def __init__(self, d_model, nhead, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
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
        gate = self.gate(normed)
        attn_out = gate * attn_out
        x = x + attn_out
        normed = self.norm2(x)
        x = x + self.ffn(normed)
        return x


class GatedResidualTransformer(nn.Module):
    """
    Gated Residual Transformer
    - Classification Head: 4-class boundary classification
    - Base Regression Head: 기본 좌표 예측
    - Residual Head: 잔차 예측
    - Gating: y_final = y_base + (1 - P_infield) * y_residual
    """
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=3, dropout=0.2, num_classes=4):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, K + 1, d_model))

        self.layers = nn.ModuleList([
            GatedTransformerBlock(d_model, nhead, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = RMSNorm(d_model)

        # Head 1: Classification (4 classes)
        self.cls_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes)
        )

        # Head 2: Base Regression (기본 좌표)
        self.base_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)
        )

        # Head 3: Residual Regression (잔차)
        self.residual_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)
        )

        # Gating parameters
        self.temperature = nn.Parameter(torch.tensor(1.0))  # learnable temperature
        self.gate_alpha = nn.Parameter(torch.tensor(1.0))   # learnable gate strength

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
        logits = self.cls_head(cls_out)
        base_coords = self.base_head(cls_out)
        residual = self.residual_head(cls_out)

        # Gating with temperature scaling
        # temperature가 낮을수록 확률이 더 극단적 (확신 있는 예측)
        proba = F.softmax(logits / self.temperature, dim=1)
        p_infield = proba[:, 0:1]  # shape: [batch, 1]
        gate = 1 - p_infield  # boundary일수록 gate 높음

        # Final coords = base + alpha * gate * residual
        gated_coords = base_coords + self.gate_alpha * gate * residual

        if return_all:
            return logits, base_coords, residual, gated_coords, gate
        return logits, gated_coords


def create_boundary_labels(end_x, end_y):
    labels = np.zeros(len(end_x), dtype=int)
    labels[(end_x > FIELD_X - 5) | (end_x < 5)] = 3
    labels[(labels == 0) & (end_y > FIELD_Y - 5)] = 1
    labels[(labels == 0) & (end_y < 5)] = 2
    return labels


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


def augment_data_multitask(X, y_class, y_coords, feature_indices):
    """Y-flip 증강"""
    X_all = [X.copy()]
    y_class_all = [y_class.copy()]
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

    y_class_yflip = y_class.copy()
    y_class_yflip[y_class == 1] = 2
    y_class_yflip[y_class == 2] = 1

    y_coords_yflip = y_coords.copy()
    y_coords_yflip[:, 1] = FIELD_Y - y_coords_yflip[:, 1]

    X_all.append(X_yflip)
    y_class_all.append(y_class_yflip)
    y_coords_all.append(y_coords_yflip)

    return np.concatenate(X_all), np.concatenate(y_class_all), np.concatenate(y_coords_all)


def main():
    log("=" * 60)
    log("Gated Residual Transformer")
    log("y_final = y_base + (1 - P_infield) * y_residual")
    log("=" * 60)

    # 데이터 로드
    log("\n[1] 데이터 로드...")
    df_all = pd.read_csv(DATA_DIR / 'train_features_v2.csv')
    X_seq, feature_indices = prepare_sequence_data(df_all, K)

    y_class = create_boundary_labels(df_all['target_end_x'].values, df_all['target_end_y'].values)
    y_coords = df_all[['target_end_x', 'target_end_y']].values

    train_idx, val_idx = train_test_split(range(len(df_all)), test_size=0.2, random_state=42)
    X_train_orig, X_val = X_seq[train_idx], X_seq[val_idx]
    y_class_train_orig, y_class_val = y_class[train_idx], y_class[val_idx]
    y_coords_train_orig, y_coords_val = y_coords[train_idx], y_coords[val_idx]

    log(f"  Train: {len(X_train_orig)}, Val: {len(X_val)}")

    log("\n[클래스 분포]")
    for i, name in enumerate(['In-field', 'Top-out', 'Bottom-out', 'Goal-line']):
        count = (y_class_train_orig == i).sum()
        log(f"  {name}: {count} ({count/len(y_class_train_orig)*100:.1f}%)")

    # 증강
    log("\n[2] 데이터 증강 (Y-flip, 2배)...")
    X_train, y_class_train, y_coords_train = augment_data_multitask(
        X_train_orig, y_class_train_orig, y_coords_train_orig, feature_indices
    )
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
    log("\n[3] Gated Residual Transformer 학습...")

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_class_train_t = torch.LongTensor(y_class_train).to(device)
    y_coords_train_t = torch.FloatTensor(y_coords_train_norm).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_class_val_t = torch.LongTensor(y_class_val).to(device)
    y_coords_val_t = torch.FloatTensor(y_coords_val_norm).to(device)

    train_dataset = TensorDataset(X_train_t, y_class_train_t, y_coords_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = GatedResidualTransformer(
        input_size=n_features,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.3,
        num_classes=4
    ).to(device)

    log(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss weights
    w_cls = 1.0
    w_base = 5.0    # Base regression
    w_gated = 10.0  # Gated regression (더 중요)
    log(f"  Loss: {w_cls}*CE + {w_base}*MSE(base) + {w_gated}*MSE(gated)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    best_dist = float('inf')
    best_state = None
    patience, patience_counter = 25, 0

    for epoch in range(150):
        model.train()
        total_loss, total_ce, total_base, total_gated = 0, 0, 0, 0

        for X_batch, y_class_batch, y_coords_batch in train_loader:
            optimizer.zero_grad()

            logits, base_coords, residual, gated_coords, gate = model(X_batch, return_all=True)

            ce_loss = F.cross_entropy(logits, y_class_batch, label_smoothing=0.1)
            base_mse = F.mse_loss(base_coords, y_coords_batch)
            gated_mse = F.mse_loss(gated_coords, y_coords_batch)

            loss = w_cls * ce_loss + w_base * base_mse + w_gated * gated_mse

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_base += base_mse.item()
            total_gated += gated_mse.item()

        model.eval()
        with torch.no_grad():
            logits, base_coords, residual, gated_coords, gate = model(X_val_t, return_all=True)
            val_preds = logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_class_val, val_preds)

            # Gated prediction distance
            gated_pred_orig = gated_coords.cpu().numpy() * coord_std + coord_mean
            gated_dist = np.sqrt(((gated_pred_orig - y_coords_val) ** 2).sum(axis=1)).mean()

            # Base prediction distance
            base_pred_orig = base_coords.cpu().numpy() * coord_std + coord_mean
            base_dist = np.sqrt(((base_pred_orig - y_coords_val) ** 2).sum(axis=1)).mean()

        scheduler.step(gated_dist)

        if gated_dist < best_dist:
            best_dist = gated_dist
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            n_batches = len(train_loader)
            log(f"  Epoch {epoch+1:3d}: CE={total_ce/n_batches:.4f}, "
                f"Base={base_dist:.2f}m, Gated={gated_dist:.2f}m, Acc={val_acc:.4f}")

        if patience_counter >= patience:
            log(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    log(f"\n  Best Gated Distance: {best_dist:.4f}m")

    # 최종 평가
    log("\n[4] 최종 평가...")
    model.eval()
    with torch.no_grad():
        logits, base_coords, residual, gated_coords, gate = model(X_val_t, return_all=True)
        proba = F.softmax(logits, dim=1).cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()

        base_pred = base_coords.cpu().numpy() * coord_std + coord_mean
        gated_pred = gated_coords.cpu().numpy() * coord_std + coord_mean
        gate_vals = gate.cpu().numpy()

    # Classification Report
    log("\n[Classification Report]")
    label_names = ['In-field', 'Top-out', 'Bottom-out', 'Goal-line']
    print(classification_report(y_class_val, preds, target_names=label_names))

    # Regression Performance
    base_dist = np.sqrt(((base_pred - y_coords_val) ** 2).sum(axis=1)).mean()
    gated_dist = np.sqrt(((gated_pred - y_coords_val) ** 2).sum(axis=1)).mean()

    log("\n[Regression Performance]")
    log(f"  Base Model Distance:   {base_dist:.4f}m")
    log(f"  Gated Model Distance:  {gated_dist:.4f}m")
    log(f"  Improvement:           {base_dist - gated_dist:.4f}m")

    # 클래스별 성능 분석
    log("\n[클래스별 Euclidean Distance]")
    for i, name in enumerate(label_names):
        mask = y_class_val == i
        if mask.sum() > 0:
            base_d = np.sqrt(((base_pred[mask] - y_coords_val[mask]) ** 2).sum(axis=1)).mean()
            gated_d = np.sqrt(((gated_pred[mask] - y_coords_val[mask]) ** 2).sum(axis=1)).mean()
            avg_gate = gate_vals[mask].mean()
            log(f"  {name:12s}: Base={base_d:.2f}m, Gated={gated_d:.2f}m, "
                f"Δ={base_d-gated_d:+.2f}m, AvgGate={avg_gate:.3f}")

    # Gate 통계
    log("\n[Gate Statistics]")
    log(f"  Mean Gate: {gate_vals.mean():.4f}")
    log(f"  Std Gate:  {gate_vals.std():.4f}")
    log(f"  Min Gate:  {gate_vals.min():.4f}")
    log(f"  Max Gate:  {gate_vals.max():.4f}")

    # Learned parameters
    log("\n[Learned Gating Parameters]")
    log(f"  Temperature: {model.temperature.item():.4f}")
    log(f"  Gate Alpha:  {model.gate_alpha.item():.4f}")

    # 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'coord_mean': coord_mean,
        'coord_std': coord_std,
        'best_dist': best_dist
    }, DATA_DIR / 'transformer_gated.pt')
    log(f"\n저장: {DATA_DIR / 'transformer_gated.pt'}")

    log("\n" + "=" * 60)
    log("완료!")
    log("=" * 60)


if __name__ == '__main__':
    main()
