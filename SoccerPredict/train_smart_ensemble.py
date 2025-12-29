"""
Smart Ensemble: 조건부 앙상블 + 분산 확대 + 경계 클리핑
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

DATA_DIR = Path('/workspace/SoccerPredict/open_track1')
FIELD_X = 105
FIELD_Y = 68
Y_CENTER = 34
X_CENTER = 52.5


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


def strategy_a_conditional_ensemble(pred_unified_y, pred_fail_y, is_fail, w_center=0.9, w_side=0.5):
    """전략 A: 조건부 앙상블 - 사이드에서 Fail 모델 가중치 높이기"""
    result = pred_unified_y.copy()

    # Fail 샘플에서만 적용
    fail_mask = is_fail

    # 사이드 여부 판단 (Unified 예측 기준)
    is_side = (pred_unified_y <= 10) | (pred_unified_y >= 58)

    # Fail + 중앙: w=0.9
    center_mask = fail_mask & ~is_side
    result[center_mask] = w_center * pred_unified_y[center_mask] + (1 - w_center) * pred_fail_y[center_mask]

    # Fail + 사이드: w=0.5
    side_mask = fail_mask & is_side
    result[side_mask] = w_side * pred_unified_y[side_mask] + (1 - w_side) * pred_fail_y[side_mask]

    return result


def strategy_b_variance_expansion(pred_y, alpha=1.1, center=Y_CENTER):
    """전략 B: 분산 확대 - 평균에서 멀리 밀어내기"""
    return center + (pred_y - center) * alpha


def strategy_c_hard_clipping(pred_x, pred_y, is_fail, y_threshold=2, x_threshold=2):
    """전략 C: 경계값 클리핑 - Fail에서 경계 근처면 경계로 붙이기"""
    new_x = pred_x.copy()
    new_y = pred_y.copy()

    # Fail 샘플에서만 적용
    fail_mask = is_fail

    # Y 클리핑
    new_y[(fail_mask) & (pred_y < y_threshold)] = 0
    new_y[(fail_mask) & (pred_y > FIELD_Y - y_threshold)] = FIELD_Y

    # X 클리핑
    new_x[(fail_mask) & (pred_x > FIELD_X - x_threshold)] = FIELD_X
    new_x[(fail_mask) & (pred_x < x_threshold)] = 0

    return new_x, new_y


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

    train_success = df_train[df_train['last_result_name'] == 'Successful']
    train_fail = df_train[df_train['last_result_name'] != 'Successful']

    log(f"Train: {len(df_train)}, Val: {len(df_val)}")

    # Train models
    log("Training models...")
    model_s_x = train_xgb(train_success[base_features].fillna(0), train_success['target_end_x'].values)
    model_s_y = train_xgb(train_success[base_features].fillna(0), train_success['target_end_y'].values)
    model_f_x = train_xgb(train_fail[base_features].fillna(0), train_fail['target_end_x'].values)
    model_f_y = train_xgb(train_fail[base_features].fillna(0), train_fail['target_end_y'].values)
    model_u_x = train_xgb(df_train[unified_features].fillna(0), df_train['target_end_x'].values)
    model_u_y = train_xgb(df_train[unified_features].fillna(0), df_train['target_end_y'].values)
    log("Done!")

    # Predictions
    X_val_base = df_val[base_features].fillna(0)
    X_val_unified = df_val[unified_features].fillna(0)

    is_success = df_val['last_result_name'] == 'Successful'
    is_fail = ~is_success

    true_x = df_val['target_end_x'].values
    true_y = df_val['target_end_y'].values

    # Base predictions
    pred_success_x = model_s_x.predict(X_val_base)
    pred_success_y = model_s_y.predict(X_val_base)
    pred_fail_x = model_f_x.predict(X_val_base)
    pred_fail_y = model_f_y.predict(X_val_base)
    pred_unified_x = model_u_x.predict(X_val_unified)
    pred_unified_y = model_u_y.predict(X_val_unified)

    # ============================================
    # 베이스라인 성능
    # ============================================
    log("\n" + "=" * 70)
    log("베이스라인 성능")
    log("=" * 70)

    # Split 모델 (기존 방식)
    pred_split_x = np.where(is_success, pred_success_x, pred_fail_x)
    pred_split_y = np.where(is_success, pred_success_y, pred_fail_y)
    pred_split_x = np.clip(pred_split_x, 0, FIELD_X)
    pred_split_y = np.clip(pred_split_y, 0, FIELD_Y)

    # Unified 모델
    pred_unified_x_clip = np.clip(pred_unified_x, 0, FIELD_X)
    pred_unified_y_clip = np.clip(pred_unified_y, 0, FIELD_Y)

    dist_split = euclidean_distance(true_x, true_y, pred_split_x, pred_split_y).mean()
    dist_unified = euclidean_distance(true_x, true_y, pred_unified_x_clip, pred_unified_y_clip).mean()

    log(f"Split 모델:   {dist_split:.4f}")
    log(f"Unified 모델: {dist_unified:.4f}")

    # ============================================
    # 전략 테스트
    # ============================================
    log("\n" + "=" * 70)
    log("전략별 성능 테스트")
    log("=" * 70)

    results = []

    # 전략 A: 조건부 앙상블 (다양한 가중치)
    log("\n[전략 A: 조건부 앙상블]")
    for w_center in [0.9, 0.8, 0.7]:
        for w_side in [0.5, 0.4, 0.3]:
            # Y에만 적용
            new_y = strategy_a_conditional_ensemble(
                pred_unified_y_clip.copy(),
                np.clip(pred_fail_y, 0, FIELD_Y),
                is_fail.values,
                w_center=w_center,
                w_side=w_side
            )
            new_x = pred_unified_x_clip.copy()  # X는 그대로

            dist = euclidean_distance(true_x, true_y, new_x, new_y).mean()
            improvement = dist_unified - dist
            results.append(('A', f'w_c={w_center},w_s={w_side}', dist, improvement))
            if improvement > 0:
                log(f"  w_center={w_center}, w_side={w_side}: {dist:.4f} (개선: {improvement:+.4f})")

    # 전략 B: 분산 확대 (다양한 alpha)
    log("\n[전략 B: 분산 확대]")
    for alpha in [1.05, 1.1, 1.15, 1.2]:
        new_y = strategy_b_variance_expansion(pred_unified_y_clip.copy(), alpha=alpha)
        new_y = np.clip(new_y, 0, FIELD_Y)
        new_x = pred_unified_x_clip.copy()

        dist = euclidean_distance(true_x, true_y, new_x, new_y).mean()
        improvement = dist_unified - dist
        results.append(('B', f'alpha={alpha}', dist, improvement))
        log(f"  alpha={alpha}: {dist:.4f} (개선: {improvement:+.4f})")

    # 전략 B를 Fail에만 적용
    log("\n[전략 B: Fail에만 분산 확대]")
    for alpha in [1.1, 1.15, 1.2, 1.3]:
        new_y = pred_unified_y_clip.copy()
        new_y[is_fail] = strategy_b_variance_expansion(new_y[is_fail], alpha=alpha)
        new_y = np.clip(new_y, 0, FIELD_Y)
        new_x = pred_unified_x_clip.copy()

        dist = euclidean_distance(true_x, true_y, new_x, new_y).mean()
        improvement = dist_unified - dist
        results.append(('B_fail', f'alpha={alpha}', dist, improvement))
        log(f"  alpha={alpha} (Fail only): {dist:.4f} (개선: {improvement:+.4f})")

    # 전략 C: 경계 클리핑 (다양한 threshold)
    log("\n[전략 C: 경계 클리핑]")
    for y_th in [2, 3, 5]:
        for x_th in [2, 3, 5]:
            new_x, new_y = strategy_c_hard_clipping(
                pred_unified_x_clip.copy(),
                pred_unified_y_clip.copy(),
                is_fail.values,
                y_threshold=y_th,
                x_threshold=x_th
            )

            dist = euclidean_distance(true_x, true_y, new_x, new_y).mean()
            improvement = dist_unified - dist
            results.append(('C', f'y_th={y_th},x_th={x_th}', dist, improvement))
            if improvement > 0:
                log(f"  y_th={y_th}, x_th={x_th}: {dist:.4f} (개선: {improvement:+.4f})")

    # ============================================
    # 복합 전략 (A + B + C)
    # ============================================
    log("\n" + "=" * 70)
    log("복합 전략 테스트")
    log("=" * 70)

    # 최고 조합 찾기
    best_configs = [
        # (A params, B alpha, C params)
        ((0.9, 0.5), 1.1, (3, 3)),
        ((0.8, 0.4), 1.15, (2, 2)),
        ((0.7, 0.3), 1.2, (3, 3)),
    ]

    for (w_c, w_s), alpha, (y_th, x_th) in best_configs:
        # Step 1: 조건부 앙상블 (A)
        new_y = strategy_a_conditional_ensemble(
            pred_unified_y_clip.copy(),
            np.clip(pred_fail_y, 0, FIELD_Y),
            is_fail.values,
            w_center=w_c,
            w_side=w_s
        )
        new_x = pred_unified_x_clip.copy()

        # Step 2: Fail에 분산 확대 (B)
        new_y[is_fail] = strategy_b_variance_expansion(new_y[is_fail], alpha=alpha)
        new_y = np.clip(new_y, 0, FIELD_Y)

        # Step 3: 경계 클리핑 (C)
        new_x, new_y = strategy_c_hard_clipping(new_x, new_y, is_fail.values, y_th, x_th)

        dist = euclidean_distance(true_x, true_y, new_x, new_y).mean()
        improvement = dist_unified - dist
        log(f"A({w_c},{w_s}) + B({alpha}) + C({y_th},{x_th}): {dist:.4f} (개선: {improvement:+.4f})")

    # ============================================
    # 최종 비교
    # ============================================
    log("\n" + "=" * 70)
    log("최종 비교")
    log("=" * 70)

    # 가장 좋은 단일 전략
    best_single = min(results, key=lambda x: x[2])
    log(f"\n최고 단일 전략: {best_single[0]} - {best_single[1]}")
    log(f"  Distance: {best_single[2]:.4f} (개선: {best_single[3]:+.4f})")

    log(f"\n베이스라인 대비:")
    log(f"  Split:   {dist_split:.4f}")
    log(f"  Unified: {dist_unified:.4f}")
    log(f"  Best:    {best_single[2]:.4f}")


if __name__ == '__main__':
    main()
