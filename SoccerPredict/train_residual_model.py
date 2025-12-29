"""
Residual Learning 구조: 분류기 + 회귀 모델 (OOF 완전 적용)

핵심 수정사항:
1. Base Model 예측값도 OOF로 생성 (Data Leakage 방지)
2. 분류기와 회귀기를 동일한 Fold 내에서 학습 (통합 OOF 루프)
3. Gating Mechanism: y_final = y_base + (1 - P_infield) * y_residual
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/workspace/SoccerPredict/open_track1')
FIELD_X = 105
FIELD_Y = 68


def log(msg):
    print(msg, flush=True)


def create_boundary_labels(end_x, end_y, threshold=5):
    """경계 라벨 생성 (0: In-field, 1: Top, 2: Bottom, 3: Goal-line)"""
    labels = np.zeros(len(end_x), dtype=int)
    labels[(end_x > FIELD_X - threshold) | (end_x < threshold)] = 3
    labels[(labels == 0) & (end_y > FIELD_Y - threshold)] = 1
    labels[(labels == 0) & (end_y < threshold)] = 2
    return labels


def get_top_k_features(X, y_x, y_y, feature_names, k=30):
    """XGBoost 피처 중요도 기반 Top K 피처 선택"""
    model_x = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42,
                                n_jobs=-1, tree_method='hist', device='cuda')
    model_x.fit(X, y_x, verbose=False)

    model_y = xgb.XGBRegressor(n_estimators=100, max_depth=6, random_state=42,
                                n_jobs=-1, tree_method='hist', device='cuda')
    model_y.fit(X, y_y, verbose=False)

    combined_imp = model_x.feature_importances_ + model_y.feature_importances_
    feat_imp = sorted(zip(feature_names, combined_imp), key=lambda x: -x[1])

    return [f for f, _ in feat_imp[:k]]


# =============================================================================
# 통합 OOF 루프: 분류기 + 베이스 회귀기를 동일 Fold에서 학습
# =============================================================================

def generate_all_oof_features(X_all, X_top30, y_x, y_y, y_labels, n_splits=5):
    """
    통합 OOF 루프: Data Leakage 완전 방지

    동일한 Fold 내에서:
    1. Base Regressor 학습 → OOF base prediction
    2. Classifier 학습 → OOF class probability

    Returns:
        oof_base_x, oof_base_y: Base 모델의 OOF 예측값
        oof_proba: 분류기의 OOF 확률
    """
    n_samples = len(X_all)
    n_classes = len(np.unique(y_labels))

    oof_base_x = np.zeros(n_samples)
    oof_base_y = np.zeros(n_samples)
    oof_proba = np.zeros((n_samples, n_classes))

    # 모델 설정
    reg_cfg = dict(
        objective='reg:squarederror',
        tree_method='hist',
        device='cuda',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    clf_cfg = dict(
        objective='multi:softprob',
        num_class=n_classes,
        tree_method='hist',
        device='cuda',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_labels)):
        # Train/Val 분리
        X_tr_all, X_val_all = X_all[train_idx], X_all[val_idx]
        X_tr_top30, X_val_top30 = X_top30[train_idx], X_top30[val_idx]
        y_tr_x, y_val_x = y_x[train_idx], y_x[val_idx]
        y_tr_y, y_val_y = y_y[train_idx], y_y[val_idx]
        y_tr_labels, y_val_labels = y_labels[train_idx], y_labels[val_idx]

        # 1. Base Regressor 학습 (Top 30 피처)
        m_x = xgb.XGBRegressor(**reg_cfg)
        m_x.fit(X_tr_top30, y_tr_x, verbose=False)
        oof_base_x[val_idx] = m_x.predict(X_val_top30)

        m_y = xgb.XGBRegressor(**reg_cfg)
        m_y.fit(X_tr_top30, y_tr_y, verbose=False)
        oof_base_y[val_idx] = m_y.predict(X_val_top30)

        # 2. Classifier 학습 (전체 피처)
        clf = xgb.XGBClassifier(**clf_cfg)
        clf.fit(X_tr_all, y_tr_labels, verbose=False)
        oof_proba[val_idx] = clf.predict_proba(X_val_all)

        # Fold 성능 출력
        base_dist = np.sqrt((y_val_x - oof_base_x[val_idx])**2 +
                           (y_val_y - oof_base_y[val_idx])**2).mean()
        clf_acc = (clf.predict(X_val_all) == y_val_labels).mean()

        log(f"    Fold {fold+1}: Base Dist = {base_dist:.2f}m, Clf Acc = {clf_acc:.4f}")

    return oof_base_x, oof_base_y, oof_proba


def main():
    log("=" * 70)
    log("Residual Learning v2: OOF 완전 적용 + Gating Mechanism")
    log("=" * 70)

    # =========================================================================
    # 1. 데이터 로드
    # =========================================================================
    log("\n[1] 데이터 로드...")
    df = pd.read_csv(DATA_DIR / 'train_features_v2.csv')

    exclude = ['game_episode', 'target_end_x', 'target_end_y',
               'last_result_name', 'last_result_encoded']
    all_features = [c for c in df.columns if c not in exclude
                    and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    log(f"전체: {len(df)} rows, {len(all_features)} features")

    # Train/Val 분리
    train_idx, val_idx = train_test_split(range(len(df)), test_size=0.2, random_state=42)
    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_val = df.iloc[val_idx].copy().reset_index(drop=True)

    y_train_x = df_train['target_end_x'].values
    y_train_y = df_train['target_end_y'].values
    y_val_x = df_val['target_end_x'].values
    y_val_y = df_val['target_end_y'].values

    # 경계 라벨
    train_labels = create_boundary_labels(y_train_x, y_train_y)
    val_labels = create_boundary_labels(y_val_x, y_val_y)

    log(f"Train: {len(df_train)}, Val: {len(df_val)}")

    # =========================================================================
    # 2. Top 30 피처 선택
    # =========================================================================
    log("\n[2] Top 30 피처 선택...")
    X_train_all = df_train[all_features].fillna(0).values
    X_val_all = df_val[all_features].fillna(0).values

    top30_cols = get_top_k_features(X_train_all, y_train_x, y_train_y, all_features, k=30)
    log(f"Top 30 피처: {top30_cols[:5]} ...")

    X_train_top30 = df_train[top30_cols].fillna(0).values
    X_val_top30 = df_val[top30_cols].fillna(0).values

    # =========================================================================
    # 3. 통합 OOF 루프 (Data Leakage 완전 방지)
    # =========================================================================
    log("\n[3] 통합 OOF 루프 (Base + Classifier)...")
    oof_base_x, oof_base_y, oof_proba = generate_all_oof_features(
        X_train_all, X_train_top30, y_train_x, y_train_y, train_labels, n_splits=5
    )

    # OOF 잔차 계산 (이제 Data Leakage 없음!)
    oof_residual_x = y_train_x - oof_base_x
    oof_residual_y = y_train_y - oof_base_y

    log(f"\nOOF Residual Stats:")
    log(f"  X: mean={oof_residual_x.mean():.2f}, std={oof_residual_x.std():.2f}")
    log(f"  Y: mean={oof_residual_y.mean():.2f}, std={oof_residual_y.std():.2f}")

    # OOF Base 성능
    oof_base_x_clip = np.clip(oof_base_x, 0, FIELD_X)
    oof_base_y_clip = np.clip(oof_base_y, 0, FIELD_Y)
    oof_base_dist = np.sqrt((y_train_x - oof_base_x_clip)**2 +
                            (y_train_y - oof_base_y_clip)**2).mean()
    log(f"  OOF Base Distance: {oof_base_dist:.4f} m")

    # =========================================================================
    # 4. Val 예측용 Full 모델 학습
    # =========================================================================
    log("\n[4] Full 모델 학습 (Val 예측용)...")

    reg_cfg = dict(
        objective='reg:squarederror',
        tree_method='hist',
        device='cuda',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )

    # Full Base Model
    base_model_x = xgb.XGBRegressor(**reg_cfg)
    base_model_x.fit(X_train_top30, y_train_x, verbose=False)
    base_pred_val_x = base_model_x.predict(X_val_top30)

    base_model_y = xgb.XGBRegressor(**reg_cfg)
    base_model_y.fit(X_train_top30, y_train_y, verbose=False)
    base_pred_val_y = base_model_y.predict(X_val_top30)

    # Full Classifier
    clf_full = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        tree_method='hist',
        device='cuda',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    clf_full.fit(X_train_all, train_labels, verbose=False)
    val_proba = clf_full.predict_proba(X_val_all)
    val_pred_labels = clf_full.predict(X_val_all)

    # Base 성능
    base_pred_val_x_clip = np.clip(base_pred_val_x, 0, FIELD_X)
    base_pred_val_y_clip = np.clip(base_pred_val_y, 0, FIELD_Y)
    base_dist = np.sqrt((y_val_x - base_pred_val_x_clip)**2 +
                        (y_val_y - base_pred_val_y_clip)**2).mean()
    log(f"Val Base Distance: {base_dist:.4f} m")

    val_acc = (val_pred_labels == val_labels).mean()
    log(f"Val Classification Acc: {val_acc:.4f}")

    # 분류 리포트
    label_names = ['In-field', 'Top-out', 'Bottom-out', 'Goal-line']
    log("\n분류 리포트:")
    print(classification_report(val_labels, val_pred_labels, target_names=label_names))

    # =========================================================================
    # 5. Residual Model 학습 (OOF 피처 사용)
    # =========================================================================
    log("\n[5] Residual Model 학습...")

    # Train 피처: Top 30 + OOF 확률 + OOF Base 예측
    X_train_res = np.hstack([
        X_train_top30,
        oof_proba,
        oof_base_x.reshape(-1, 1),
        oof_base_y.reshape(-1, 1)
    ])

    # Val 피처: Top 30 + Full 확률 + Full Base 예측
    X_val_res = np.hstack([
        X_val_top30,
        val_proba,
        base_pred_val_x.reshape(-1, 1),
        base_pred_val_y.reshape(-1, 1)
    ])

    log(f"Residual 피처: Top30(30) + 분류확률(4) + Base예측(2) = {X_train_res.shape[1]}개")

    # 잔차 정규화 (안정적 학습)
    scaler_res_x = StandardScaler()
    scaler_res_y = StandardScaler()
    oof_residual_x_scaled = scaler_res_x.fit_transform(oof_residual_x.reshape(-1, 1)).ravel()
    oof_residual_y_scaled = scaler_res_y.fit_transform(oof_residual_y.reshape(-1, 1)).ravel()

    # Residual 모델 (Huber Loss 대신 기본 MSE 사용, scale로 안정화)
    res_model_x = xgb.XGBRegressor(**reg_cfg)
    res_model_x.fit(X_train_res, oof_residual_x_scaled, verbose=False)

    res_model_y = xgb.XGBRegressor(**reg_cfg)
    res_model_y.fit(X_train_res, oof_residual_y_scaled, verbose=False)

    # Val 잔차 예측 (역정규화)
    res_pred_val_x_scaled = res_model_x.predict(X_val_res)
    res_pred_val_y_scaled = res_model_y.predict(X_val_res)
    res_pred_val_x = scaler_res_x.inverse_transform(res_pred_val_x_scaled.reshape(-1, 1)).ravel()
    res_pred_val_y = scaler_res_y.inverse_transform(res_pred_val_y_scaled.reshape(-1, 1)).ravel()

    # =========================================================================
    # 6. 최종 예측: Gating Mechanism
    # y_final = y_base + (1 - P_infield) * y_residual
    # =========================================================================
    log("\n[6] 최종 예측: Gating Mechanism...")

    # Gating: In-field 확률이 높으면 잔차 보정 줄이기
    gate = 1 - val_proba[:, 0]  # 1 - P(in-field)

    # 방법 1: 단순 합산 (Gating 없음)
    final_pred_x_simple = base_pred_val_x + res_pred_val_x
    final_pred_y_simple = base_pred_val_y + res_pred_val_y
    final_pred_x_simple = np.clip(final_pred_x_simple, 0, FIELD_X)
    final_pred_y_simple = np.clip(final_pred_y_simple, 0, FIELD_Y)
    dist_simple = np.sqrt((y_val_x - final_pred_x_simple)**2 +
                          (y_val_y - final_pred_y_simple)**2).mean()

    # 방법 2: Gating 적용
    final_pred_x_gated = base_pred_val_x + gate * res_pred_val_x
    final_pred_y_gated = base_pred_val_y + gate * res_pred_val_y
    final_pred_x_gated = np.clip(final_pred_x_gated, 0, FIELD_X)
    final_pred_y_gated = np.clip(final_pred_y_gated, 0, FIELD_Y)
    dist_gated = np.sqrt((y_val_x - final_pred_x_gated)**2 +
                         (y_val_y - final_pred_y_gated)**2).mean()

    # 방법 3: 다양한 Gating 강도
    log("\n  Gating 강도별 성능:")
    for alpha in [0.5, 0.7, 1.0, 1.3, 1.5]:
        gate_scaled = np.clip((1 - val_proba[:, 0]) * alpha, 0, 1)
        pred_x = base_pred_val_x + gate_scaled * res_pred_val_x
        pred_y = base_pred_val_y + gate_scaled * res_pred_val_y
        pred_x = np.clip(pred_x, 0, FIELD_X)
        pred_y = np.clip(pred_y, 0, FIELD_Y)
        dist = np.sqrt((y_val_x - pred_x)**2 + (y_val_y - pred_y)**2).mean()
        log(f"    alpha={alpha}: {dist:.4f} m")

    # =========================================================================
    # 7. 결과 비교
    # =========================================================================
    log("\n" + "=" * 70)
    log("결과 비교")
    log("=" * 70)
    log(f"| 모델                              | Mean Distance | 개선       |")
    log(f"|-----------------------------------|---------------|------------|")
    log(f"| Base Model (Top 30)               | {base_dist:.4f} m     | -          |")
    log(f"| Base + Residual (단순합)          | {dist_simple:.4f} m     | {dist_simple - base_dist:+.4f} m   |")
    log(f"| Base + Residual (Gating)          | {dist_gated:.4f} m     | {dist_gated - base_dist:+.4f} m   |")

    # Residual 피처 중요도
    log("\n[7] Residual Model 피처 중요도:")
    feature_names = top30_cols + ['prob_infield', 'prob_top', 'prob_bottom', 'prob_goal',
                                   'oof_base_x', 'oof_base_y']

    imp_x = res_model_x.feature_importances_
    feat_imp = sorted(zip(feature_names, imp_x), key=lambda x: -x[1])

    for i, (f, imp) in enumerate(feat_imp[:15], 1):
        marker = " ***" if f.startswith('prob_') or f.startswith('oof_') else ""
        log(f"  {i:2d}. {f:30s} {imp:.4f}{marker}")

    # 분류별 성능 분석
    log("\n[8] 클래스별 성능 (Gating 적용):")
    for i, name in enumerate(label_names):
        mask = val_labels == i
        if mask.sum() == 0:
            continue

        base_d = np.sqrt((y_val_x[mask] - base_pred_val_x_clip[mask])**2 +
                         (y_val_y[mask] - base_pred_val_y_clip[mask])**2).mean()
        gated_d = np.sqrt((y_val_x[mask] - final_pred_x_gated[mask])**2 +
                          (y_val_y[mask] - final_pred_y_gated[mask])**2).mean()

        log(f"  {name:12s}: Base={base_d:.2f}m → Gated={gated_d:.2f}m ({gated_d - base_d:+.2f}m)")

    return base_model_x, base_model_y, res_model_x, res_model_y, clf_full


if __name__ == '__main__':
    models = main()
