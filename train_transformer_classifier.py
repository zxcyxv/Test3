"""
Transformer 경계 분류기
- 4-class: In-field, Top-out, Bottom-out, Goal-line
- 4x 데이터 증강
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
import math
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
    """
    Pre-Norm Transformer Block with RMSNorm + Gated Attention
    Gate(X) ⊙ SDPA(Q, K, V) - Attention Sink 제거, 표현력 향상
    """
    def __init__(self, d_model, nhead, dropout=0.2):
        super().__init__()
        self.d_model = d_model

        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Gate: Linear + Sigmoid (input-dependent sparsity)
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
        # Pre-norm Gated Attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)

        # Gating: Gate(X) ⊙ Attention_Output
        gate = self.gate(normed)
        attn_out = gate * attn_out

        x = x + attn_out

        # Pre-norm FFN
        normed = self.norm2(x)
        x = x + self.ffn(normed)
        return x


class MultiTaskTransformer(nn.Module):
    """
    Multi-Task Transformer with [CLS] Token + RMSNorm
    - 분류: 4-class boundary classification
    - 회귀: end_x, end_y 좌표 예측 (물리적 위치 감각 학습용)
    """
    def __init__(self, input_size, d_model=128, nhead=4, num_layers=3, dropout=0.2, num_classes=4):
        super().__init__()

        # [CLS] 토큰: 시퀀스 전체의 정보를 요약할 학습 가능한 벡터
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Input Projection
        self.input_proj = nn.Linear(input_size, d_model)

        # Positional Encoding (학습 가능)
        self.pos_embedding = nn.Parameter(torch.randn(1, K + 1, d_model))  # K seq + 1 CLS

        # Transformer Encoder with RMSNorm + Gated Attention
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

        # Head 2: Regression (Predict Normalized End X, End Y)
        self.reg_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 2)  # (x, y)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 1. Projection
        x = self.input_proj(x)  # [batch, seq, d_model]

        # 2. Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch, seq+1, d_model]

        # 3. Add Position Embedding
        x = x + self.pos_embedding[:, :x.size(1), :]

        # 4. Transformer Encoding
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)

        # 5. Take [CLS] token output (Global Context)
        cls_out = x[:, 0, :]

        # 6. Multi-Task Output
        logits = self.cls_head(cls_out)
        coords = self.reg_head(cls_out)

        return logits, coords


def create_boundary_labels(end_x, end_y):
    labels = np.zeros(len(end_x), dtype=int)
    labels[(end_x > FIELD_X - 5) | (end_x < 5)] = 3
    labels[(labels == 0) & (end_y > FIELD_Y - 5)] = 1
    labels[(labels == 0) & (end_y < 5)] = 2
    return labels


def prepare_sequence_data(df, k=K):
    # 기본 피처 (모든 K 이벤트에 존재)
    base_features = ['start_x', 'start_y', 'dt', 'ep_idx_norm', 'x_zone', 'lane',
                     'dist_to_goal', 'angle_to_goal', 'type_id', 'res_id', 'is_home',
                     # V2 새 피처
                     'pressure_x_weight', 'is_zone14', 'angle_visible']
    # 마스킹 피처 (마지막 이벤트 제외)
    masked_features = ['end_x', 'end_y', 'dx', 'dy', 'dist', 'speed',
                       # V2 새 피처 (마지막 제외)
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


def augment_data(X, y, feature_indices):
    """Y-flip만 적용 (2배 증강). X-flip은 데이터가 이미 한 방향으로 정규화되어 의미 없음."""
    X_all, y_all = [X.copy()], [y.copy()]

    # Y-flip (상하 대칭)
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
    # 라벨도 Top(1) <-> Bottom(2) 교환
    y_yflip = y.copy()
    y_yflip[y == 1] = 2
    y_yflip[y == 2] = 1
    X_all.append(X_yflip)
    y_all.append(y_yflip)

    return np.concatenate(X_all), np.concatenate(y_all)


def augment_data_multitask(X, y_class, y_coords, feature_indices):
    """Y-flip 증강 (분류 라벨 + 좌표 함께)"""
    X_all = [X.copy()]
    y_class_all = [y_class.copy()]
    y_coords_all = [y_coords.copy()]

    # Y-flip
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

    # 라벨 Top(1) <-> Bottom(2)
    y_class_yflip = y_class.copy()
    y_class_yflip[y_class == 1] = 2
    y_class_yflip[y_class == 2] = 1

    # 좌표도 Y-flip
    y_coords_yflip = y_coords.copy()
    y_coords_yflip[:, 1] = FIELD_Y - y_coords_yflip[:, 1]  # end_y flip

    X_all.append(X_yflip)
    y_class_all.append(y_class_yflip)
    y_coords_all.append(y_coords_yflip)

    return np.concatenate(X_all), np.concatenate(y_class_all), np.concatenate(y_coords_all)


def main():
    log("=" * 60)
    log("Multi-Task Transformer (분류 + 회귀) with [CLS] Token")
    log("=" * 60)

    # 데이터 로드
    log("\n[1] 데이터 로드...")
    df_all = pd.read_csv(DATA_DIR / 'train_features_v2.csv')
    X_seq, feature_indices = prepare_sequence_data(df_all, K)

    # 분류 라벨 + 좌표 타겟
    y_class = create_boundary_labels(df_all['target_end_x'].values, df_all['target_end_y'].values)
    y_coords = df_all[['target_end_x', 'target_end_y']].values

    train_idx, val_idx = train_test_split(range(len(df_all)), test_size=0.2, random_state=42)
    X_train_orig, X_val = X_seq[train_idx], X_seq[val_idx]
    y_class_train_orig, y_class_val = y_class[train_idx], y_class[val_idx]
    y_coords_train_orig, y_coords_val = y_coords[train_idx], y_coords[val_idx]

    log("\n[클래스 분포]")
    for i, name in enumerate(['In-field', 'Top-out', 'Bottom-out', 'Goal-line']):
        count = (y_class_train_orig == i).sum()
        log(f"  {name}: {count} ({count/len(y_class_train_orig)*100:.1f}%)")

    # 증강
    log("\n[2] 데이터 증강 (Y-flip, 2배)...")
    X_train, y_class_train, y_coords_train = augment_data_multitask(
        X_train_orig, y_class_train_orig, y_coords_train_orig, feature_indices
    )
    log(f"    Train: {len(X_train)}, Val: {len(X_val)}")

    # 정규화
    n_samples, n_steps, n_features = X_train.shape
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_samples, n_steps, n_features)
    X_val = scaler.transform(X_val.reshape(-1, n_features)).reshape(len(X_val), n_steps, n_features)

    # 좌표 정규화 (0~1)
    coord_mean = np.array([FIELD_X / 2, FIELD_Y / 2])
    coord_std = np.array([FIELD_X / 2, FIELD_Y / 2])
    y_coords_train_norm = (y_coords_train - coord_mean) / coord_std
    y_coords_val_norm = (y_coords_val - coord_mean) / coord_std

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f"    Device: {device}")

    # 학습
    log("\n[3] Multi-Task Transformer 학습...")

    X_train_t = torch.FloatTensor(X_train).to(device)
    y_class_train_t = torch.LongTensor(y_class_train).to(device)
    y_coords_train_t = torch.FloatTensor(y_coords_train_norm).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    y_class_val_t = torch.LongTensor(y_class_val).to(device)
    y_coords_val_t = torch.FloatTensor(y_coords_val_norm).to(device)

    train_dataset = TensorDataset(X_train_t, y_class_train_t, y_coords_train_t)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = MultiTaskTransformer(
        input_size=n_features,
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.3,
        num_classes=4
    ).to(device)

    log(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Multi-Task Loss
    # w_reg를 높게: 회귀 loss 스케일이 작으므로 가중치를 높여서 좌표 감각 학습 강화
    w_cls = 1.0
    w_reg = 10.0
    log(f"    Loss: {w_cls} * CE + {w_reg} * MSE")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    best_acc, best_state = 0, None
    patience, patience_counter = 25, 0

    for epoch in range(150):
        model.train()
        total_loss, total_ce, total_mse = 0, 0, 0
        for X_batch, y_class_batch, y_coords_batch in train_loader:
            optimizer.zero_grad()

            class_logits, coord_pred = model(X_batch)

            ce_loss = F.cross_entropy(class_logits, y_class_batch, label_smoothing=0.1)
            mse_loss = F.mse_loss(coord_pred, y_coords_batch)
            loss = w_cls * ce_loss + w_reg * mse_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_mse += mse_loss.item()

        model.eval()
        with torch.no_grad():
            val_class_logits, val_coord_pred = model(X_val_t)
            val_preds = val_class_logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_class_val, val_preds)

            # 회귀 성능 (Euclidean distance)
            val_coord_pred_orig = val_coord_pred.cpu().numpy() * coord_std + coord_mean
            val_dist = np.sqrt(((val_coord_pred_orig - y_coords_val) ** 2).sum(axis=1)).mean()

        scheduler.step(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            n_batches = len(train_loader)
            log(f"    Epoch {epoch+1}: CE={total_ce/n_batches:.4f}, MSE={total_mse/n_batches:.4f}, Val Acc={val_acc:.4f}, Val Dist={val_dist:.2f}")

        if patience_counter >= patience:
            log(f"    Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    log(f"\n    Best Accuracy: {best_acc:.4f}")

    # 평가
    model.eval()
    with torch.no_grad():
        class_logits, coord_pred = model(X_val_t)
        proba = torch.softmax(class_logits, dim=1).cpu().numpy()
        preds = class_logits.argmax(dim=1).cpu().numpy()
        coord_pred_orig = coord_pred.cpu().numpy() * coord_std + coord_mean

    log("\n[4] Classification Report")
    label_names = ['In-field', 'Top-out', 'Bottom-out', 'Goal-line']
    print(classification_report(y_class_val, preds, target_names=label_names))

    log("\n[Confusion Matrix]")
    cm = confusion_matrix(y_class_val, preds)
    for i, name in enumerate(label_names):
        log(f"  True {name:12s}: {cm[i]}")

    # In-field / Boundary Recall
    infield_mask = y_class_val == 0
    boundary_mask = y_class_val > 0
    log(f"\n  In-field Recall: {(preds[infield_mask] == 0).mean():.4f}")
    log(f"  Boundary Recall: {(preds[boundary_mask] > 0).mean():.4f}")

    # 회귀 성능
    val_dist = np.sqrt(((coord_pred_orig - y_coords_val) ** 2).sum(axis=1)).mean()
    log(f"  Regression Euclidean Distance: {val_dist:.2f}")

    # 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
        'coord_mean': coord_mean,
        'coord_std': coord_std,
        'accuracy': best_acc
    }, DATA_DIR / 'transformer_multitask.pt')
    log(f"\n저장: {DATA_DIR / 'transformer_multitask.pt'}")


if __name__ == '__main__':
    main()
