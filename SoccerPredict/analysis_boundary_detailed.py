"""
경계 근처 모델 성능 상세 분석: Success/Fail/Unified 각각
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

    # Train/Val masks
    train_success = df_train[df_train['last_result_name'] == 'Successful']
    train_fail = df_train[df_train['last_result_name'] != 'Successful']
    val_success_mask = df_val['last_result_name'] == 'Successful'
    val_fail_mask = ~val_success_mask

    val_success = df_val[val_success_mask].copy().reset_index(drop=True)
    val_fail = df_val[val_fail_mask].copy().reset_index(drop=True)

    log(f"Train - Success: {len(train_success)}, Fail: {len(train_fail)}")
    log(f"Val - Success: {len(val_success)}, Fail: {len(val_fail)}")

    # Train models
    log("\nTraining models...")

    # Success model
    model_s_x = train_xgb(train_success[base_features].fillna(0), train_success['target_end_x'].values)
    model_s_y = train_xgb(train_success[base_features].fillna(0), train_success['target_end_y'].values)
    log("  Success model done")

    # Fail model
    model_f_x = train_xgb(train_fail[base_features].fillna(0), train_fail['target_end_x'].values)
    model_f_y = train_xgb(train_fail[base_features].fillna(0), train_fail['target_end_y'].values)
    log("  Fail model done")

    # Unified model
    model_u_x = train_xgb(df_train[unified_features].fillna(0), df_train['target_end_x'].values)
    model_u_y = train_xgb(df_train[unified_features].fillna(0), df_train['target_end_y'].values)
    log("  Unified model done")

    # ============================================
    # Success 샘플 분석 (Success 모델 vs Unified)
    # ============================================
    log("\n" + "=" * 70)
    log("SUCCESS 샘플 분석 (n={})".format(len(val_success)))
    log("=" * 70)

    X_s = val_success[base_features].fillna(0)
    X_s_unified = val_success[unified_features].fillna(0)
    true_x_s = val_success['target_end_x'].values
    true_y_s = val_success['target_end_y'].values

    pred_success_x = np.clip(model_s_x.predict(X_s), 0, FIELD_X)
    pred_success_y = np.clip(model_s_y.predict(X_s), 0, FIELD_Y)
    pred_unified_s_x = np.clip(model_u_x.predict(X_s_unified), 0, FIELD_X)
    pred_unified_s_y = np.clip(model_u_y.predict(X_s_unified), 0, FIELD_Y)

    dist_success = euclidean_distance(true_x_s, true_y_s, pred_success_x, pred_success_y)
    dist_unified_s = euclidean_distance(true_x_s, true_y_s, pred_unified_s_x, pred_unified_s_y)

    log(f"\n전체 평균: Success모델={dist_success.mean():.2f}, Unified={dist_unified_s.mean():.2f}")

    # Y 구간별
    log("\n[Y축 구간별]")
    y_regions = [
        ("Y < 5", true_y_s < 5),
        ("5 <= Y < 15", (true_y_s >= 5) & (true_y_s < 15)),
        ("15 <= Y < 34", (true_y_s >= 15) & (true_y_s < 34)),
        ("34 <= Y < 53", (true_y_s >= 34) & (true_y_s < 53)),
        ("53 <= Y < 63", (true_y_s >= 53) & (true_y_s < 63)),
        ("Y >= 63", true_y_s >= 63),
    ]

    for name, mask in y_regions:
        n = mask.sum()
        if n > 0:
            s_dist = dist_success[mask].mean()
            u_dist = dist_unified_s[mask].mean()
            diff = s_dist - u_dist
            winner = "Success" if s_dist < u_dist else "Unified"
            log(f"  {name:20s}: n={n:4d}, Success모델={s_dist:.2f}, Unified={u_dist:.2f}, 차이={diff:+.2f} -> {winner}")

    # X 구간별
    log("\n[X축 구간별]")
    x_regions = [
        ("X < 20", true_x_s < 20),
        ("20 <= X < 52.5", (true_x_s >= 20) & (true_x_s < 52.5)),
        ("52.5 <= X < 85", (true_x_s >= 52.5) & (true_x_s < 85)),
        ("X >= 85", true_x_s >= 85),
    ]

    for name, mask in x_regions:
        n = mask.sum()
        if n > 0:
            s_dist = dist_success[mask].mean()
            u_dist = dist_unified_s[mask].mean()
            diff = s_dist - u_dist
            winner = "Success" if s_dist < u_dist else "Unified"
            log(f"  {name:20s}: n={n:4d}, Success모델={s_dist:.2f}, Unified={u_dist:.2f}, 차이={diff:+.2f} -> {winner}")

    # ============================================
    # Fail 샘플 분석 (Fail 모델 vs Unified)
    # ============================================
    log("\n" + "=" * 70)
    log("FAIL 샘플 분석 (n={})".format(len(val_fail)))
    log("=" * 70)

    X_f = val_fail[base_features].fillna(0)
    X_f_unified = val_fail[unified_features].fillna(0)
    true_x_f = val_fail['target_end_x'].values
    true_y_f = val_fail['target_end_y'].values

    pred_fail_x = np.clip(model_f_x.predict(X_f), 0, FIELD_X)
    pred_fail_y = np.clip(model_f_y.predict(X_f), 0, FIELD_Y)
    pred_unified_f_x = np.clip(model_u_x.predict(X_f_unified), 0, FIELD_X)
    pred_unified_f_y = np.clip(model_u_y.predict(X_f_unified), 0, FIELD_Y)

    dist_fail = euclidean_distance(true_x_f, true_y_f, pred_fail_x, pred_fail_y)
    dist_unified_f = euclidean_distance(true_x_f, true_y_f, pred_unified_f_x, pred_unified_f_y)

    log(f"\n전체 평균: Fail모델={dist_fail.mean():.2f}, Unified={dist_unified_f.mean():.2f}")

    # Y 구간별
    log("\n[Y축 구간별]")
    y_regions_f = [
        ("Y < 5", true_y_f < 5),
        ("5 <= Y < 15", (true_y_f >= 5) & (true_y_f < 15)),
        ("15 <= Y < 34", (true_y_f >= 15) & (true_y_f < 34)),
        ("34 <= Y < 53", (true_y_f >= 34) & (true_y_f < 53)),
        ("53 <= Y < 63", (true_y_f >= 53) & (true_y_f < 63)),
        ("Y >= 63", true_y_f >= 63),
    ]

    for name, mask in y_regions_f:
        n = mask.sum()
        if n > 0:
            f_dist = dist_fail[mask].mean()
            u_dist = dist_unified_f[mask].mean()
            diff = f_dist - u_dist
            winner = "Fail" if f_dist < u_dist else "Unified"
            log(f"  {name:20s}: n={n:4d}, Fail모델={f_dist:.2f}, Unified={u_dist:.2f}, 차이={diff:+.2f} -> {winner}")

    # X 구간별
    log("\n[X축 구간별]")
    x_regions_f = [
        ("X < 20", true_x_f < 20),
        ("20 <= X < 52.5", (true_x_f >= 20) & (true_x_f < 52.5)),
        ("52.5 <= X < 85", (true_x_f >= 52.5) & (true_x_f < 85)),
        ("X >= 85", true_x_f >= 85),
    ]

    for name, mask in x_regions_f:
        n = mask.sum()
        if n > 0:
            f_dist = dist_fail[mask].mean()
            u_dist = dist_unified_f[mask].mean()
            diff = f_dist - u_dist
            winner = "Fail" if f_dist < u_dist else "Unified"
            log(f"  {name:20s}: n={n:4d}, Fail모델={f_dist:.2f}, Unified={u_dist:.2f}, 차이={diff:+.2f} -> {winner}")

    # ============================================
    # 경계 예측 능력 비교
    # ============================================
    log("\n" + "=" * 70)
    log("경계 예측 능력 비교")
    log("=" * 70)

    log("\n[Success 샘플 - Y 경계 Recall]")
    for threshold in [5, 10]:
        true_low = true_y_s < threshold
        true_high = true_y_s > (68 - threshold)

        if true_low.sum() > 0:
            s_recall = (true_low & (pred_success_y < threshold)).sum() / true_low.sum() * 100
            u_recall = (true_low & (pred_unified_s_y < threshold)).sum() / true_low.sum() * 100
            log(f"  Y<{threshold}: Success모델={s_recall:.1f}%, Unified={u_recall:.1f}%")

        if true_high.sum() > 0:
            s_recall = (true_high & (pred_success_y > (68-threshold))).sum() / true_high.sum() * 100
            u_recall = (true_high & (pred_unified_s_y > (68-threshold))).sum() / true_high.sum() * 100
            log(f"  Y>{68-threshold}: Success모델={s_recall:.1f}%, Unified={u_recall:.1f}%")

    log("\n[Fail 샘플 - Y 경계 Recall]")
    for threshold in [5, 10]:
        true_low = true_y_f < threshold
        true_high = true_y_f > (68 - threshold)

        if true_low.sum() > 0:
            f_recall = (true_low & (pred_fail_y < threshold)).sum() / true_low.sum() * 100
            u_recall = (true_low & (pred_unified_f_y < threshold)).sum() / true_low.sum() * 100
            log(f"  Y<{threshold}: Fail모델={f_recall:.1f}%, Unified={u_recall:.1f}%")

        if true_high.sum() > 0:
            f_recall = (true_high & (pred_fail_y > (68-threshold))).sum() / true_high.sum() * 100
            u_recall = (true_high & (pred_unified_f_y > (68-threshold))).sum() / true_high.sum() * 100
            log(f"  Y>{68-threshold}: Fail모델={f_recall:.1f}%, Unified={u_recall:.1f}%")

    # ============================================
    # 예측값 분포 통계
    # ============================================
    log("\n" + "=" * 70)
    log("예측값 분포 통계")
    log("=" * 70)

    log("\n[Success 샘플]")
    log(f"  True Y:        min={true_y_s.min():.1f}, max={true_y_s.max():.1f}, std={true_y_s.std():.1f}")
    log(f"  Success모델 Y: min={pred_success_y.min():.1f}, max={pred_success_y.max():.1f}, std={pred_success_y.std():.1f}")
    log(f"  Unified Y:     min={pred_unified_s_y.min():.1f}, max={pred_unified_s_y.max():.1f}, std={pred_unified_s_y.std():.1f}")

    log("\n[Fail 샘플]")
    log(f"  True Y:        min={true_y_f.min():.1f}, max={true_y_f.max():.1f}, std={true_y_f.std():.1f}")
    log(f"  Fail모델 Y:    min={pred_fail_y.min():.1f}, max={pred_fail_y.max():.1f}, std={pred_fail_y.std():.1f}")
    log(f"  Unified Y:     min={pred_unified_f_y.min():.1f}, max={pred_unified_f_y.max():.1f}, std={pred_unified_f_y.std():.1f}")

    # ============================================
    # 요약
    # ============================================
    log("\n" + "=" * 70)
    log("요약")
    log("=" * 70)

    # Success 샘플에서 어디가 더 좋은지
    s_better_regions = []
    u_better_regions_s = []
    for name, mask in y_regions:
        n = mask.sum()
        if n > 10:
            s_dist = dist_success[mask].mean()
            u_dist = dist_unified_s[mask].mean()
            if s_dist < u_dist:
                s_better_regions.append(f"{name}({s_dist:.1f}<{u_dist:.1f})")
            else:
                u_better_regions_s.append(f"{name}({u_dist:.1f}<{s_dist:.1f})")

    log(f"\nSuccess 샘플에서:")
    log(f"  Success모델 우세: {', '.join(s_better_regions) if s_better_regions else 'None'}")
    log(f"  Unified 우세: {', '.join(u_better_regions_s) if u_better_regions_s else 'None'}")

    # Fail 샘플에서 어디가 더 좋은지
    f_better_regions = []
    u_better_regions_f = []
    for name, mask in y_regions_f:
        n = mask.sum()
        if n > 10:
            f_dist = dist_fail[mask].mean()
            u_dist = dist_unified_f[mask].mean()
            if f_dist < u_dist:
                f_better_regions.append(f"{name}({f_dist:.1f}<{u_dist:.1f})")
            else:
                u_better_regions_f.append(f"{name}({u_dist:.1f}<{f_dist:.1f})")

    log(f"\nFail 샘플에서:")
    log(f"  Fail모델 우세: {', '.join(f_better_regions) if f_better_regions else 'None'}")
    log(f"  Unified 우세: {', '.join(u_better_regions_f) if u_better_regions_f else 'None'}")


if __name__ == '__main__':
    main()
