"""
오차 분포 분석: Split, Unified 모델 비교
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


def train_xgb(X_train, y_train, X_val=None, y_val=None):
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'verbosity': 0,
    }
    if X_val is not None:
        params['early_stopping_rounds'] = 20
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
    return model


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

    # Split masks
    val_success_mask = df_val['last_result_name'] == 'Successful'
    val_fail_mask = ~val_success_mask
    train_success = df_train[df_train['last_result_name'] == 'Successful']
    train_fail = df_train[df_train['last_result_name'] != 'Successful']

    # Ground truth
    y_x_val = df_val['target_end_x'].values
    y_y_val = df_val['target_end_y'].values

    # ============================================
    # Train models and get predictions
    # ============================================
    log("Training models...")

    # Split Success model
    X_s_tr = train_success[base_features].fillna(0)
    model_s_x = train_xgb(X_s_tr, train_success['target_end_x'].values)
    model_s_y = train_xgb(X_s_tr, train_success['target_end_y'].values)

    # Split Fail model
    X_f_tr = train_fail[base_features].fillna(0)
    model_f_x = train_xgb(X_f_tr, train_fail['target_end_x'].values)
    model_f_y = train_xgb(X_f_tr, train_fail['target_end_y'].values)

    # Unified model
    X_tr = df_train[unified_features].fillna(0)
    model_u_x = train_xgb(X_tr, df_train['target_end_x'].values)
    model_u_y = train_xgb(X_tr, df_train['target_end_y'].values)

    log("Generating predictions...")

    # Split predictions (각각의 모델로)
    X_val_base = df_val[base_features].fillna(0)
    X_val_unified = df_val[unified_features].fillna(0)

    pred_split_x = np.zeros(len(df_val))
    pred_split_y = np.zeros(len(df_val))
    pred_split_x[val_success_mask] = model_s_x.predict(X_val_base[val_success_mask])
    pred_split_y[val_success_mask] = model_s_y.predict(X_val_base[val_success_mask])
    pred_split_x[val_fail_mask] = model_f_x.predict(X_val_base[val_fail_mask])
    pred_split_y[val_fail_mask] = model_f_y.predict(X_val_base[val_fail_mask])

    # Unified predictions
    pred_unified_x = model_u_x.predict(X_val_unified)
    pred_unified_y = model_u_y.predict(X_val_unified)

    # Clip
    pred_split_x = np.clip(pred_split_x, 0, FIELD_X)
    pred_split_y = np.clip(pred_split_y, 0, FIELD_Y)
    pred_unified_x = np.clip(pred_unified_x, 0, FIELD_X)
    pred_unified_y = np.clip(pred_unified_y, 0, FIELD_Y)

    # ============================================
    # Calculate errors
    # ============================================
    log("\n" + "=" * 60)
    log("오차 분포 분석")
    log("=" * 60)

    # Errors
    err_split_x = pred_split_x - y_x_val
    err_split_y = pred_split_y - y_y_val
    err_unified_x = pred_unified_x - y_x_val
    err_unified_y = pred_unified_y - y_y_val

    # Euclidean distance
    dist_split = np.sqrt(err_split_x**2 + err_split_y**2)
    dist_unified = np.sqrt(err_unified_x**2 + err_unified_y**2)

    # ============================================
    # 1. 기본 통계
    # ============================================
    log("\n[1] 기본 통계")
    log("-" * 40)

    for name, err_x, err_y, dist in [
        ("Split", err_split_x, err_split_y, dist_split),
        ("Unified", err_unified_x, err_unified_y, dist_unified)
    ]:
        log(f"\n{name} Model:")
        log(f"  X 오차: mean={err_x.mean():.3f}, std={err_x.std():.3f}, |mean|={np.abs(err_x).mean():.3f}")
        log(f"  Y 오차: mean={err_y.mean():.3f}, std={err_y.std():.3f}, |mean|={np.abs(err_y).mean():.3f}")
        log(f"  거리:   mean={dist.mean():.3f}, std={dist.std():.3f}, median={np.median(dist):.3f}")
        log(f"  거리 분위: 25%={np.percentile(dist, 25):.2f}, 75%={np.percentile(dist, 75):.2f}, 90%={np.percentile(dist, 90):.2f}")

    # ============================================
    # 2. Success vs Fail 별 분석
    # ============================================
    log("\n[2] Success vs Fail 별 분석")
    log("-" * 40)

    for mask, label in [(val_success_mask, "Success"), (val_fail_mask, "Fail")]:
        log(f"\n{label} 샘플 (n={mask.sum()}):")

        for name, err_x, err_y, dist in [
            ("Split", err_split_x[mask], err_split_y[mask], dist_split[mask]),
            ("Unified", err_unified_x[mask], err_unified_y[mask], dist_unified[mask])
        ]:
            log(f"  {name}: X_err={err_x.mean():+.2f}, Y_err={err_y.mean():+.2f}, dist={dist.mean():.2f}")

    # ============================================
    # 3. 과대/과소 예측 경향
    # ============================================
    log("\n[3] 과대/과소 예측 경향")
    log("-" * 40)

    for name, err_x, err_y in [
        ("Split", err_split_x, err_split_y),
        ("Unified", err_unified_x, err_unified_y)
    ]:
        over_x = (err_x > 0).mean() * 100
        over_y = (err_y > 0).mean() * 100
        log(f"{name}: X 과대예측 {over_x:.1f}%, Y 과대예측 {over_y:.1f}%")

    # ============================================
    # 4. 큰 오차 케이스 분석
    # ============================================
    log("\n[4] 큰 오차 케이스 분석 (거리 > 30m)")
    log("-" * 40)

    for name, dist, mask_s in [
        ("Split", dist_split, val_success_mask),
        ("Unified", dist_unified, val_success_mask)
    ]:
        large_err = dist > 30
        n_large = large_err.sum()
        pct = n_large / len(dist) * 100

        success_in_large = (large_err & mask_s).sum()
        fail_in_large = (large_err & ~mask_s).sum()

        log(f"{name}: {n_large}개 ({pct:.1f}%) - Success: {success_in_large}, Fail: {fail_in_large}")

    # ============================================
    # 5. 모델 간 비교 (같은 샘플에서)
    # ============================================
    log("\n[5] 모델 간 비교")
    log("-" * 40)

    split_better = dist_split < dist_unified
    log(f"Split이 더 나은 샘플: {split_better.sum()} ({split_better.mean()*100:.1f}%)")
    log(f"Unified가 더 나은 샘플: {(~split_better).sum()} ({(~split_better).mean()*100:.1f}%)")

    # Success/Fail 별
    log(f"\nSuccess 샘플에서:")
    s_better_in_success = split_better[val_success_mask].mean() * 100
    log(f"  Split이 더 나은 비율: {s_better_in_success:.1f}%")

    log(f"Fail 샘플에서:")
    s_better_in_fail = split_better[val_fail_mask].mean() * 100
    log(f"  Split이 더 나은 비율: {s_better_in_fail:.1f}%")

    # ============================================
    # 6. 시각화
    # ============================================
    log("\n[6] 시각화 저장 중...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    # Row 1: X 오차 분포
    axes[0, 0].hist(err_split_x, bins=50, alpha=0.7, label='Split', color='blue')
    axes[0, 0].hist(err_unified_x, bins=50, alpha=0.7, label='Unified', color='orange')
    axes[0, 0].axvline(0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('X Error (pred - true)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('X 좌표 오차 분포')
    axes[0, 0].legend()

    # Row 1: Y 오차 분포
    axes[0, 1].hist(err_split_y, bins=50, alpha=0.7, label='Split', color='blue')
    axes[0, 1].hist(err_unified_y, bins=50, alpha=0.7, label='Unified', color='orange')
    axes[0, 1].axvline(0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Y Error (pred - true)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Y 좌표 오차 분포')
    axes[0, 1].legend()

    # Row 2: 유클리드 거리 분포
    axes[1, 0].hist(dist_split, bins=50, alpha=0.7, label='Split', color='blue')
    axes[1, 0].hist(dist_unified, bins=50, alpha=0.7, label='Unified', color='orange')
    axes[1, 0].set_xlabel('Euclidean Distance (m)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('유클리드 거리 오차 분포')
    axes[1, 0].legend()

    # Row 2: Success vs Fail 거리 비교
    data_to_plot = [
        dist_split[val_success_mask], dist_unified[val_success_mask],
        dist_split[val_fail_mask], dist_unified[val_fail_mask]
    ]
    labels = ['Split\nSuccess', 'Unified\nSuccess', 'Split\nFail', 'Unified\nFail']
    bp = axes[1, 1].boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['lightblue', 'lightyellow', 'lightblue', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    axes[1, 1].set_ylabel('Euclidean Distance (m)')
    axes[1, 1].set_title('Success vs Fail 별 오차 분포')

    # Row 3: 실제 vs 예측 산점도
    sample_idx = np.random.choice(len(df_val), min(500, len(df_val)), replace=False)

    axes[2, 0].scatter(y_x_val[sample_idx], pred_split_x[sample_idx], alpha=0.3, s=10, label='Split')
    axes[2, 0].scatter(y_x_val[sample_idx], pred_unified_x[sample_idx], alpha=0.3, s=10, label='Unified')
    axes[2, 0].plot([0, FIELD_X], [0, FIELD_X], 'r--', label='Perfect')
    axes[2, 0].set_xlabel('True X')
    axes[2, 0].set_ylabel('Predicted X')
    axes[2, 0].set_title('실제 vs 예측 X좌표')
    axes[2, 0].legend()

    axes[2, 1].scatter(y_y_val[sample_idx], pred_split_y[sample_idx], alpha=0.3, s=10, label='Split')
    axes[2, 1].scatter(y_y_val[sample_idx], pred_unified_y[sample_idx], alpha=0.3, s=10, label='Unified')
    axes[2, 1].plot([0, FIELD_Y], [0, FIELD_Y], 'r--', label='Perfect')
    axes[2, 1].set_xlabel('True Y')
    axes[2, 1].set_ylabel('Predicted Y')
    axes[2, 1].set_title('실제 vs 예측 Y좌표')
    axes[2, 1].legend()

    plt.tight_layout()
    plt.savefig(DATA_DIR / 'error_distribution.png', dpi=150)
    log(f"  저장 완료: {DATA_DIR / 'error_distribution.png'}")


if __name__ == '__main__':
    main()
