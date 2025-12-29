"""
Zone-wise Feature Analysis
- 영역별 별도 모델 학습 후 Feature Importance 비교
- Rank Shift: 영역별 피처 중요도 순위 변화
- Magnitude Explosion: 경계 영역에서 영향력 증폭
- Prediction Error Pattern: 영역별 오차 분포
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

DATA_DIR = Path('/workspace/SoccerPredict/open_track1')
FIELD_X, FIELD_Y = 105, 68


def create_boundary_labels(end_x, end_y):
    """경계 라벨 생성"""
    labels = np.zeros(len(end_x), dtype=int)
    labels[(end_x > FIELD_X - 5) | (end_x < 5)] = 3  # Goal-line
    labels[(labels == 0) & (end_y > FIELD_Y - 5)] = 1  # Top-out
    labels[(labels == 0) & (end_y < 5)] = 2  # Bottom-out
    return labels


def main():
    log("=" * 70)
    log("Zone-wise Feature Importance Analysis")
    log("영역별 모델 학습 후 Feature Importance 비교")
    log("=" * 70)

    # 1. 데이터 로드
    log("\n[1] 데이터 로드...")
    df = pd.read_csv(DATA_DIR / 'train_features_v2.csv')

    # 타겟 분리 (숫자형 피처만 사용)
    target_cols = ['target_end_x', 'target_end_y']
    drop_cols = target_cols + ['game_episode']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # 숫자형 컬럼만 필터링
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = numeric_cols

    X = df[feature_cols].values
    y_x = df['target_end_x'].values
    y_y = df['target_end_y'].values
    y_labels = create_boundary_labels(df['target_end_x'].values, df['target_end_y'].values)

    log(f"  Total samples: {len(df)}")
    log(f"  Features: {len(feature_cols)}")

    # 영역별 샘플 수
    label_names = ['In-field', 'Top-out', 'Bottom-out', 'Goal-line']
    log("\n[영역별 샘플 분포]")
    for i, name in enumerate(label_names):
        count = (y_labels == i).sum()
        log(f"  {name}: {count} ({count/len(y_labels)*100:.1f}%)")

    # 2. 전체 모델 학습
    log("\n[2] 전체 데이터로 XGBoost 학습...")
    X_train, X_val, y_x_train, y_x_val, y_labels_train, y_labels_val = train_test_split(
        X, y_x, y_labels, test_size=0.2, random_state=42
    )

    model_all = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    model_all.fit(X_train, y_x_train, eval_set=[(X_val, y_x_val)], verbose=False)

    # 전체 모델 Feature Importance
    imp_all = pd.Series(model_all.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # 3. 영역별 모델 학습
    log("\n[3] 영역별 XGBoost 학습...")
    zone_models = {}
    zone_importance = {}

    for zone_id, zone_name in enumerate(label_names):
        mask_train = y_labels_train == zone_id
        mask_val = y_labels_val == zone_id

        if mask_train.sum() < 100:
            log(f"  {zone_name}: 샘플 부족 ({mask_train.sum()}), 스킵")
            continue

        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        )
        model.fit(
            X_train[mask_train], y_x_train[mask_train],
            eval_set=[(X_val[mask_val], y_x_val[mask_val])] if mask_val.sum() > 0 else None,
            verbose=False
        )

        zone_models[zone_name] = model
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        zone_importance[zone_name] = imp

        # 예측 성능
        if mask_val.sum() > 0:
            pred = model.predict(X_val[mask_val])
            rmse = np.sqrt(np.mean((pred - y_x_val[mask_val]) ** 2))
            log(f"  {zone_name}: RMSE={rmse:.2f}m (n_train={mask_train.sum()}, n_val={mask_val.sum()})")

    # 4. Feature Importance 비교
    log("\n" + "=" * 70)
    log("[4] Zone-wise Feature Importance Analysis")
    log("=" * 70)

    # A. Top 15 Features per Zone
    log("\n[A] Top 15 Features per Zone")
    log("-" * 70)

    log("\n  [전체 모델]")
    for rank, (feat, val) in enumerate(imp_all.head(15).items(), 1):
        log(f"    {rank:2d}. {feat:35s}: {val:.4f}")

    for zone_name, imp in zone_importance.items():
        log(f"\n  [{zone_name}]")
        for rank, (feat, val) in enumerate(imp.head(15).items(), 1):
            log(f"    {rank:2d}. {feat:35s}: {val:.4f}")

    # B. Rank Shift 분석: In-field vs Goal-line
    log("\n" + "=" * 70)
    log("[B] Rank Shift Analysis: In-field vs Goal-line")
    log("=" * 70)

    if 'In-field' in zone_importance and 'Goal-line' in zone_importance:
        imp_if = zone_importance['In-field']
        imp_gl = zone_importance['Goal-line']

        # 상위 30 피처의 순위 변화
        all_top_features = set(imp_if.head(30).index) | set(imp_gl.head(30).index)

        log("\n  {:35s} | {:>8s} | {:>8s} | {:>8s} | {:>8s}".format(
            "Feature", "IF Rank", "GL Rank", "Rank Δ", "Imp Δ"
        ))
        log("-" * 80)

        rank_shifts = []
        for feat in all_top_features:
            rank_if = list(imp_if.index).index(feat) + 1
            rank_gl = list(imp_gl.index).index(feat) + 1
            imp_diff = imp_gl[feat] - imp_if[feat]
            shift = rank_if - rank_gl
            rank_shifts.append((feat, rank_if, rank_gl, shift, imp_diff))

        # 순위 변화가 큰 순서로 정렬
        rank_shifts.sort(key=lambda x: abs(x[3]), reverse=True)
        for feat, rank_if, rank_gl, shift, imp_diff in rank_shifts[:25]:
            log(f"  {feat:35s} | {rank_if:8d} | {rank_gl:8d} | {shift:+8d} | {imp_diff:+8.4f}")

        # 가장 큰 순위 변화 요약
        log("\n  [가장 큰 순위 변화 Top 10]")
        for feat, rank_if, rank_gl, shift, imp_diff in rank_shifts[:10]:
            if shift > 0:
                direction = "↑ Goal-line에서 더 중요"
            else:
                direction = "↓ In-field에서 더 중요"
            log(f"    {feat:35s}: {shift:+d} ({direction})")

    # C. Magnitude Explosion 분석
    log("\n" + "=" * 70)
    log("[C] Magnitude Explosion Analysis")
    log("=" * 70)

    if 'In-field' in zone_importance and 'Goal-line' in zone_importance:
        imp_if = zone_importance['In-field']
        imp_gl = zone_importance['Goal-line']

        mag_changes = []
        for feat in feature_cols:
            val_if = imp_if[feat]
            val_gl = imp_gl[feat]
            if val_if > 0.001:
                ratio = val_gl / val_if
                mag_changes.append((feat, val_if, val_gl, ratio))

        # 증폭 비율이 높은 순서
        mag_changes.sort(key=lambda x: x[3], reverse=True)

        log("\n  [Goal-line에서 영향력 증폭 Top 15] (ratio > 1)")
        log("  {:35s} | {:>10s} | {:>10s} | {:>8s}".format("Feature", "In-field", "Goal-line", "Ratio"))
        log("-" * 70)
        for feat, val_if, val_gl, ratio in mag_changes[:15]:
            log(f"  {feat:35s} | {val_if:10.4f} | {val_gl:10.4f} | {ratio:8.2f}x")

        log("\n  [Goal-line에서 영향력 감소 Top 15] (ratio < 1)")
        mag_changes.sort(key=lambda x: x[3])
        log("  {:35s} | {:>10s} | {:>10s} | {:>8s}".format("Feature", "In-field", "Goal-line", "Ratio"))
        log("-" * 70)
        for feat, val_if, val_gl, ratio in mag_changes[:15]:
            log(f"  {feat:35s} | {val_if:10.4f} | {val_gl:10.4f} | {ratio:8.2f}x")

    # D. 피처 카테고리별 영역 간 중요도 비교
    log("\n" + "=" * 70)
    log("[D] Feature Category Analysis")
    log("=" * 70)

    # 피처 카테고리 정의
    categories = {
        'Coordinate': [c for c in feature_cols if any(x in c for x in ['start_x', 'start_y', 'end_x', 'end_y', 'last_start'])],
        'Distance/Angle': [c for c in feature_cols if any(x in c for x in ['dist_to_goal', 'angle', 'dist_'])],
        'Sequence': [c for c in feature_cols if any(x in c for x in ['action_', 'prev_', 'recent_'])],
        'Zone/Position': [c for c in feature_cols if any(x in c for x in ['zone', 'lane', 'pressure', 'field_third'])],
        'Type/Result': [c for c in feature_cols if any(x in c for x in ['type_id', 'res_id', 'is_home'])],
    }

    if 'In-field' in zone_importance and 'Goal-line' in zone_importance:
        log("\n  {:20s} | {:>12s} | {:>12s} | {:>8s}".format("Category", "In-field", "Goal-line", "Ratio"))
        log("-" * 60)

        for cat_name, cat_features in categories.items():
            if cat_features:
                imp_if_cat = zone_importance['In-field'][cat_features].sum()
                imp_gl_cat = zone_importance['Goal-line'][cat_features].sum()
                ratio = imp_gl_cat / imp_if_cat if imp_if_cat > 0.001 else 0
                log(f"  {cat_name:20s} | {imp_if_cat:12.4f} | {imp_gl_cat:12.4f} | {ratio:8.2f}x")

    # E. Winning Signal 판정
    log("\n" + "=" * 70)
    log("[E] Winning Signal 판정")
    log("=" * 70)

    if 'In-field' in zone_importance and 'Goal-line' in zone_importance:
        # Signal 1: Rank Shift (순위 ≥10 변화)
        significant_shifts = sum(1 for _, _, _, s, _ in rank_shifts if abs(s) >= 10)
        log(f"\n  [1] Rank Shift (순위 ≥10 변화): {significant_shifts}개 피처")
        if significant_shifts >= 5:
            log("      ✓ 영역별 피처 중요도 구조가 확연히 다름")
            log("      → FiLM/MoE 등 영역 조건부 모델링 근거 확보")
        else:
            log("      △ 영역 간 중요도 구조 유사")

        # Signal 2: Magnitude Explosion (2x 이상 증폭)
        high_explosion = sum(1 for _, _, _, r in mag_changes if r >= 2.0)
        log(f"\n  [2] Magnitude Explosion (2x 이상 증폭): {high_explosion}개 피처")
        if high_explosion >= 5:
            log("      ✓ 경계 영역에서 특정 피처 영향력 폭증")
            log("      → 영역별 전문화된 Head 필요")
        else:
            log("      △ 증폭 현상 미미")

        # Signal 3: 카테고리별 반전
        coord_ratio = zone_importance['Goal-line'][categories['Coordinate']].sum() / \
                      zone_importance['In-field'][categories['Coordinate']].sum()
        seq_ratio = zone_importance['Goal-line'][categories['Sequence']].sum() / \
                    zone_importance['In-field'][categories['Sequence']].sum()

        log(f"\n  [3] Category Shift")
        log(f"      Coordinate 피처 (GL/IF): {coord_ratio:.2f}x")
        log(f"      Sequence 피처 (GL/IF): {seq_ratio:.2f}x")
        if coord_ratio > 1.5 and seq_ratio < 0.7:
            log("      ✓ Goal-line에서 좌표 중심, In-field에서 시퀀스 중심")
            log("      → 영역별 다른 학습 전략 필요")

    log("\n" + "=" * 70)
    log("분석 완료!")
    log("=" * 70)


if __name__ == '__main__':
    main()
