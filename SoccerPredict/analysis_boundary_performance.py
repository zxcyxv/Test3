"""
경계 근처 모델 성능 분석
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from pathlib import Path
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

DATA_DIR = Path('/workspace/SoccerPredict/open_track1')
FIELD_X = 105
FIELD_Y = 68


def train_xgb(X_train, y_train):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        device='cuda',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        verbosity=0,
    )
    model.fit(X_train, y_train, verbose=False)
    return model


def euclidean_distance(y_true_x, y_true_y, y_pred_x, y_pred_y):
    return np.sqrt((y_true_x - y_pred_x)**2 + (y_true_y - y_pred_y)**2)


def main():
    log("Loading data...")
    df_all = pd.read_csv(DATA_DIR / 'train_features_k5.csv')

    exclude = ['game_episode', 'target_end_x', 'target_end_y', 'last_result_name', 'last_result_encoded']
    base_features = [c for c in df_all.columns if c not in exclude]
    unified_features = base_features + ['last_result_encoded']

    # Train/Val split
    train_idx, val_idx = train_test_split(range(len(df_all)), test_size=0.2, random_state=42)
    df_train = df_all.iloc[train_idx].copy().reset_index(drop=True)
    df_val = df_all.iloc[val_idx].copy().reset_index(drop=True)

    log(f"Train: {len(df_train)}, Val: {len(df_val)}")

    # Train models
    log("Training models...")

    # Split models
    train_success = df_train[df_train['last_result_name'] == 'Successful']
    train_fail = df_train[df_train['last_result_name'] != 'Successful']

    model_s_x = train_xgb(train_success[base_features].fillna(0), train_success['target_end_x'].values)
    model_s_y = train_xgb(train_success[base_features].fillna(0), train_success['target_end_y'].values)
    model_f_x = train_xgb(train_fail[base_features].fillna(0), train_fail['target_end_x'].values)
    model_f_y = train_xgb(train_fail[base_features].fillna(0), train_fail['target_end_y'].values)

    # Unified models
    model_u_x = train_xgb(df_train[unified_features].fillna(0), df_train['target_end_x'].values)
    model_u_y = train_xgb(df_train[unified_features].fillna(0), df_train['target_end_y'].values)

    # Predictions
    log("Generating predictions...")
    X_val_base = df_val[base_features].fillna(0)
    X_val_unified = df_val[unified_features].fillna(0)

    val_success_mask = df_val['last_result_name'] == 'Successful'

    pred_split_x = np.zeros(len(df_val))
    pred_split_y = np.zeros(len(df_val))
    pred_split_x[val_success_mask] = model_s_x.predict(X_val_base[val_success_mask])
    pred_split_y[val_success_mask] = model_s_y.predict(X_val_base[val_success_mask])
    pred_split_x[~val_success_mask] = model_f_x.predict(X_val_base[~val_success_mask])
    pred_split_y[~val_success_mask] = model_f_y.predict(X_val_base[~val_success_mask])

    pred_unified_x = model_u_x.predict(X_val_unified)
    pred_unified_y = model_u_y.predict(X_val_unified)

    # Clip
    pred_split_x = np.clip(pred_split_x, 0, FIELD_X)
    pred_split_y = np.clip(pred_split_y, 0, FIELD_Y)
    pred_unified_x = np.clip(pred_unified_x, 0, FIELD_X)
    pred_unified_y = np.clip(pred_unified_y, 0, FIELD_Y)

    # Ground truth
    true_x = df_val['target_end_x'].values
    true_y = df_val['target_end_y'].values

    # Distances
    dist_split = euclidean_distance(true_x, true_y, pred_split_x, pred_split_y)
    dist_unified = euclidean_distance(true_x, true_y, pred_unified_x, pred_unified_y)

    # ============================================
    # 경계별 성능 분석
    # ============================================
    log("\n" + "=" * 60)
    log("경계 근처 모델 성능 분석")
    log("=" * 60)

    # Y 경계 (사이드라인)
    log("\n[1] Y축 경계 (사이드라인) - true end_y 기준")
    log("-" * 50)

    y_regions = [
        ("Y < 5 (왼쪽 터치라인)", true_y < 5),
        ("5 <= Y < 20", (true_y >= 5) & (true_y < 20)),
        ("20 <= Y < 48 (중앙)", (true_y >= 20) & (true_y < 48)),
        ("48 <= Y < 63", (true_y >= 48) & (true_y < 63)),
        ("Y >= 63 (오른쪽 터치라인)", true_y >= 63),
    ]

    for name, mask in y_regions:
        n = mask.sum()
        if n > 0:
            split_dist = dist_split[mask].mean()
            unified_dist = dist_unified[mask].mean()
            winner = "Split" if split_dist < unified_dist else "Unified"
            log(f"{name}: n={n:4d}, Split={split_dist:.2f}, Unified={unified_dist:.2f} -> {winner}")

    # X 경계 (골라인)
    log("\n[2] X축 경계 (골라인) - true end_x 기준")
    log("-" * 50)

    x_regions = [
        ("X < 10 (수비 깊숙이)", true_x < 10),
        ("10 <= X < 35 (수비)", (true_x >= 10) & (true_x < 35)),
        ("35 <= X < 70 (중앙)", (true_x >= 35) & (true_x < 70)),
        ("70 <= X < 95 (공격)", (true_x >= 70) & (true_x < 95)),
        ("X >= 95 (공격 깊숙이)", true_x >= 95),
    ]

    for name, mask in x_regions:
        n = mask.sum()
        if n > 0:
            split_dist = dist_split[mask].mean()
            unified_dist = dist_unified[mask].mean()
            winner = "Split" if split_dist < unified_dist else "Unified"
            log(f"{name}: n={n:4d}, Split={split_dist:.2f}, Unified={unified_dist:.2f} -> {winner}")

    # 코너 영역
    log("\n[3] 코너/특수 영역")
    log("-" * 50)

    corner_regions = [
        ("좌하단 코너 (X<15, Y<10)", (true_x < 15) & (true_y < 10)),
        ("좌상단 코너 (X<15, Y>58)", (true_x < 15) & (true_y > 58)),
        ("우하단 코너 (X>90, Y<10)", (true_x > 90) & (true_y < 10)),
        ("우상단 코너 (X>90, Y>58)", (true_x > 90) & (true_y > 58)),
        ("페널티에어리어 (X>90, 25<Y<43)", (true_x > 90) & (true_y > 25) & (true_y < 43)),
    ]

    for name, mask in corner_regions:
        n = mask.sum()
        if n > 0:
            split_dist = dist_split[mask].mean()
            unified_dist = dist_unified[mask].mean()
            winner = "Split" if split_dist < unified_dist else "Unified"
            log(f"{name}: n={n:4d}, Split={split_dist:.2f}, Unified={unified_dist:.2f} -> {winner}")

    # Success vs Fail 별 경계 성능
    log("\n[4] Success vs Fail 별 경계 성능")
    log("-" * 50)

    for result_name, result_mask in [("Success", val_success_mask), ("Fail", ~val_success_mask)]:
        log(f"\n{result_name}:")
        for region_name, region_mask in [("Y 사이드 (<5 or >63)", (true_y < 5) | (true_y > 63)),
                                          ("Y 중앙 (20~48)", (true_y >= 20) & (true_y < 48)),
                                          ("X 공격깊숙이 (>95)", true_x >= 95)]:
            combined_mask = result_mask & region_mask
            n = combined_mask.sum()
            if n > 0:
                split_dist = dist_split[combined_mask].mean()
                unified_dist = dist_unified[combined_mask].mean()
                winner = "Split" if split_dist < unified_dist else "Unified"
                log(f"  {region_name}: n={n:3d}, Split={split_dist:.2f}, Unified={unified_dist:.2f} -> {winner}")

    # ============================================
    # 예측값 분포 분석
    # ============================================
    log("\n" + "=" * 60)
    log("[5] 예측값 경계 도달 분석")
    log("=" * 60)

    log("\n실제 경계값 vs 예측 경계값 비교:")

    log("\n[True Y 경계 (<5 or >63)인 샘플에서 예측값]")
    y_boundary_mask = (true_y < 5) | (true_y > 63)
    log(f"  샘플 수: {y_boundary_mask.sum()}")
    log(f"  실제 Y 평균: {true_y[y_boundary_mask].mean():.2f}")
    log(f"  Split 예측 Y 평균: {pred_split_y[y_boundary_mask].mean():.2f}")
    log(f"  Unified 예측 Y 평균: {pred_unified_y[y_boundary_mask].mean():.2f}")

    # 경계 예측 정확도
    log("\n[경계 예측 정확도]")
    for threshold in [5, 10]:
        true_y_low = true_y < threshold
        true_y_high = true_y > (68 - threshold)

        split_y_low = pred_split_y < threshold
        split_y_high = pred_split_y > (68 - threshold)
        unified_y_low = pred_unified_y < threshold
        unified_y_high = pred_unified_y > (68 - threshold)

        # True가 경계일 때 예측도 경계인 비율
        if true_y_low.sum() > 0:
            split_recall_low = (true_y_low & split_y_low).sum() / true_y_low.sum() * 100
            unified_recall_low = (true_y_low & unified_y_low).sum() / true_y_low.sum() * 100
            log(f"  Y<{threshold} Recall: Split={split_recall_low:.1f}%, Unified={unified_recall_low:.1f}%")

        if true_y_high.sum() > 0:
            split_recall_high = (true_y_high & split_y_high).sum() / true_y_high.sum() * 100
            unified_recall_high = (true_y_high & unified_y_high).sum() / true_y_high.sum() * 100
            log(f"  Y>{68-threshold} Recall: Split={split_recall_high:.1f}%, Unified={unified_recall_high:.1f}%")

    # ============================================
    # 시각화
    # ============================================
    log("\n시각화 생성 중...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Y 구간별 오차
    y_bins = [0, 5, 20, 48, 63, 68]
    y_labels = ['<5', '5-20', '20-48', '48-63', '>63']
    split_by_y = []
    unified_by_y = []

    for i in range(len(y_bins)-1):
        mask = (true_y >= y_bins[i]) & (true_y < y_bins[i+1])
        if i == len(y_bins)-2:  # 마지막 구간은 <= 포함
            mask = (true_y >= y_bins[i]) & (true_y <= y_bins[i+1])
        split_by_y.append(dist_split[mask].mean() if mask.sum() > 0 else 0)
        unified_by_y.append(dist_unified[mask].mean() if mask.sum() > 0 else 0)

    x_pos = np.arange(len(y_labels))
    width = 0.35
    axes[0, 0].bar(x_pos - width/2, split_by_y, width, label='Split', color='blue', alpha=0.7)
    axes[0, 0].bar(x_pos + width/2, unified_by_y, width, label='Unified', color='orange', alpha=0.7)
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(y_labels)
    axes[0, 0].set_xlabel('True end_y region')
    axes[0, 0].set_ylabel('Mean Euclidean Distance (m)')
    axes[0, 0].set_title('Y 구간별 오차')
    axes[0, 0].legend()

    # 2. X 구간별 오차
    x_bins = [0, 10, 35, 70, 95, 105]
    x_labels = ['<10', '10-35', '35-70', '70-95', '>95']
    split_by_x = []
    unified_by_x = []

    for i in range(len(x_bins)-1):
        mask = (true_x >= x_bins[i]) & (true_x < x_bins[i+1])
        if i == len(x_bins)-2:
            mask = (true_x >= x_bins[i]) & (true_x <= x_bins[i+1])
        split_by_x.append(dist_split[mask].mean() if mask.sum() > 0 else 0)
        unified_by_x.append(dist_unified[mask].mean() if mask.sum() > 0 else 0)

    axes[0, 1].bar(x_pos - width/2, split_by_x, width, label='Split', color='blue', alpha=0.7)
    axes[0, 1].bar(x_pos + width/2, unified_by_x, width, label='Unified', color='orange', alpha=0.7)
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(x_labels)
    axes[0, 1].set_xlabel('True end_x region')
    axes[0, 1].set_ylabel('Mean Euclidean Distance (m)')
    axes[0, 1].set_title('X 구간별 오차')
    axes[0, 1].legend()

    # 3. Y 경계 근처 예측 산점도
    y_boundary = (true_y < 10) | (true_y > 58)
    axes[1, 0].scatter(true_y[y_boundary], pred_split_y[y_boundary], alpha=0.3, s=10, label='Split')
    axes[1, 0].scatter(true_y[y_boundary], pred_unified_y[y_boundary], alpha=0.3, s=10, label='Unified')
    axes[1, 0].plot([0, 68], [0, 68], 'r--', label='Perfect')
    axes[1, 0].set_xlabel('True Y')
    axes[1, 0].set_ylabel('Predicted Y')
    axes[1, 0].set_title('Y 경계 근처 예측 (Y<10 or Y>58)')
    axes[1, 0].legend()

    # 4. X 경계 근처 예측 산점도
    x_boundary = (true_x < 15) | (true_x > 90)
    axes[1, 1].scatter(true_x[x_boundary], pred_split_x[x_boundary], alpha=0.3, s=10, label='Split')
    axes[1, 1].scatter(true_x[x_boundary], pred_unified_x[x_boundary], alpha=0.3, s=10, label='Unified')
    axes[1, 1].plot([0, 105], [0, 105], 'r--', label='Perfect')
    axes[1, 1].set_xlabel('True X')
    axes[1, 1].set_ylabel('Predicted X')
    axes[1, 1].set_title('X 경계 근처 예측 (X<15 or X>90)')
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(DATA_DIR / 'boundary_performance.png', dpi=150)
    log(f"저장 완료: {DATA_DIR / 'boundary_performance.png'}")


if __name__ == '__main__':
    main()
