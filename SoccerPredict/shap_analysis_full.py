"""
Zone-wise SHAP Analysis
- TreeExplainer for XGBoost SHAP values
- Rank Shift: 영역별 피처 중요도 순위 변화
- Sign Reversal: SHAP 방향 반전 (양 ↔ 음)
- Magnitude Explosion: 경계 영역에서 영향력 증폭
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
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
    log("Zone-wise SHAP Analysis")
    log("SHAP values로 Rank Shift, Sign Reversal, Magnitude Explosion 분석")
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
    y_labels = create_boundary_labels(df['target_end_x'].values, df['target_end_y'].values)

    log(f"  Total samples: {len(df)}")
    log(f"  Features: {len(feature_cols)}")

    # 영역별 샘플 수
    label_names = ['In-field', 'Top-out', 'Bottom-out', 'Goal-line']
    log("\n[영역별 샘플 분포]")
    for i, name in enumerate(label_names):
        count = (y_labels == i).sum()
        log(f"  {name}: {count} ({count/len(y_labels)*100:.1f}%)")

    # 2. XGBoost 모델 학습
    log("\n[2] XGBoost 학습...")
    X_train, X_val, y_x_train, y_x_val, y_labels_train, y_labels_val = train_test_split(
        X, y_x, y_labels, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_x_train, eval_set=[(X_val, y_x_val)], verbose=False)

    # 3. SHAP values 계산
    log("\n[3] SHAP values 계산 (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)

    # Validation set에서 계산 (메모리 절약)
    shap_values = explainer.shap_values(X_val)
    log(f"  SHAP values shape: {shap_values.shape}")

    # SHAP DataFrame 생성
    shap_df = pd.DataFrame(shap_values, columns=feature_cols)

    # 4. 영역별 Mean SHAP 분석
    log("\n" + "=" * 70)
    log("[4] Zone-wise Mean SHAP Analysis")
    log("=" * 70)

    zone_shap = {}
    for zone_id, zone_name in enumerate(label_names):
        mask = y_labels_val == zone_id
        if mask.sum() > 0:
            zone_shap[zone_name] = shap_df.loc[mask].mean()
            log(f"  {zone_name}: {mask.sum()} samples")

    # A. Top 15 Mean |SHAP| per Zone
    log("\n[A] Top 15 Mean |SHAP| per Zone")
    log("-" * 70)

    zone_abs_shap = {}
    for zone_name, shap_series in zone_shap.items():
        abs_shap = shap_series.abs().sort_values(ascending=False)
        zone_abs_shap[zone_name] = abs_shap

        log(f"\n  [{zone_name}]")
        for rank, (feat, val) in enumerate(abs_shap.head(15).items(), 1):
            sign = "+" if shap_series[feat] > 0 else "-"
            log(f"    {rank:2d}. {feat:35s}: {sign}{val:.4f}")

    # B. Sign Reversal 분석
    log("\n" + "=" * 70)
    log("[B] Sign Reversal Analysis: In-field vs Goal-line")
    log("=" * 70)

    if 'In-field' in zone_shap and 'Goal-line' in zone_shap:
        shap_if = zone_shap['In-field']
        shap_gl = zone_shap['Goal-line']

        sign_reversals = []
        for feat in feature_cols:
            val_if = shap_if[feat]
            val_gl = shap_gl[feat]

            # 부호 반전 체크 (한쪽이 0에 가까우면 제외)
            if abs(val_if) > 0.01 and abs(val_gl) > 0.01:
                if (val_if > 0 and val_gl < 0) or (val_if < 0 and val_gl > 0):
                    sign_reversals.append((feat, val_if, val_gl))

        log(f"\n  Sign Reversal 발견: {len(sign_reversals)}개 피처")
        log("\n  {:35s} | {:>12s} | {:>12s}".format("Feature", "In-field", "Goal-line"))
        log("-" * 65)

        # 영향력이 큰 순서로 정렬
        sign_reversals.sort(key=lambda x: abs(x[1]) + abs(x[2]), reverse=True)
        for feat, val_if, val_gl in sign_reversals[:20]:
            log(f"  {feat:35s} | {val_if:+12.4f} | {val_gl:+12.4f}")

    # C. Rank Shift 분석
    log("\n" + "=" * 70)
    log("[C] Rank Shift Analysis: In-field vs Goal-line")
    log("=" * 70)

    if 'In-field' in zone_abs_shap and 'Goal-line' in zone_abs_shap:
        abs_if = zone_abs_shap['In-field']
        abs_gl = zone_abs_shap['Goal-line']

        # 상위 30 피처의 순위 변화
        all_top_features = set(abs_if.head(30).index) | set(abs_gl.head(30).index)

        log("\n  {:35s} | {:>8s} | {:>8s} | {:>8s}".format(
            "Feature", "IF Rank", "GL Rank", "Rank Δ"
        ))
        log("-" * 70)

        rank_shifts = []
        for feat in all_top_features:
            rank_if = list(abs_if.index).index(feat) + 1
            rank_gl = list(abs_gl.index).index(feat) + 1
            shift = rank_if - rank_gl
            rank_shifts.append((feat, rank_if, rank_gl, shift))

        # 순위 변화가 큰 순서로 정렬
        rank_shifts.sort(key=lambda x: abs(x[3]), reverse=True)
        for feat, rank_if, rank_gl, shift in rank_shifts[:25]:
            log(f"  {feat:35s} | {rank_if:8d} | {rank_gl:8d} | {shift:+8d}")

    # D. Magnitude Explosion 분석
    log("\n" + "=" * 70)
    log("[D] Magnitude Explosion Analysis")
    log("=" * 70)

    if 'In-field' in zone_abs_shap and 'Goal-line' in zone_abs_shap:
        abs_if = zone_abs_shap['In-field']
        abs_gl = zone_abs_shap['Goal-line']

        mag_changes = []
        for feat in feature_cols:
            val_if = abs_if[feat]
            val_gl = abs_gl[feat]
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

    # E. Feature Category별 Sign 분석
    log("\n" + "=" * 70)
    log("[E] Feature Category Sign Analysis")
    log("=" * 70)

    # 피처 카테고리 정의
    categories = {
        'Coordinate': [c for c in feature_cols if any(x in c for x in ['start_x', 'start_y', 'end_x', 'end_y', 'last_start'])],
        'Distance/Angle': [c for c in feature_cols if any(x in c for x in ['dist_to_goal', 'angle', 'dist_'])],
        'Sequence': [c for c in feature_cols if any(x in c for x in ['action_', 'prev_', 'recent_'])],
        'Zone/Position': [c for c in feature_cols if any(x in c for x in ['zone', 'lane', 'pressure', 'field_third'])],
    }

    if 'In-field' in zone_shap and 'Goal-line' in zone_shap:
        log("\n  {:20s} | {:>12s} | {:>12s} | {:>12s}".format(
            "Category", "IF Mean", "GL Mean", "Sign Change?"
        ))
        log("-" * 65)

        for cat_name, cat_features in categories.items():
            if cat_features:
                mean_if = zone_shap['In-field'][cat_features].mean()
                mean_gl = zone_shap['Goal-line'][cat_features].mean()
                sign_change = "YES" if (mean_if > 0 and mean_gl < 0) or (mean_if < 0 and mean_gl > 0) else "NO"
                log(f"  {cat_name:20s} | {mean_if:+12.4f} | {mean_gl:+12.4f} | {sign_change:>12s}")

    # F. Winning Signal 판정
    log("\n" + "=" * 70)
    log("[F] Winning Signal 판정")
    log("=" * 70)

    if 'In-field' in zone_shap and 'Goal-line' in zone_shap:
        # Signal 1: Rank Shift (순위 ≥10 변화)
        significant_shifts = sum(1 for _, _, _, s in rank_shifts if abs(s) >= 10)
        log(f"\n  [1] Rank Shift (순위 ≥10 변화): {significant_shifts}개 피처")
        if significant_shifts >= 5:
            log("      ✓ 영역별 피처 중요도 구조가 확연히 다름")
            log("      → FiLM/MoE 등 영역 조건부 모델링 근거 확보")
        else:
            log("      △ 영역 간 중요도 구조 유사")

        # Signal 2: Sign Reversal
        log(f"\n  [2] Sign Reversal: {len(sign_reversals)}개 피처")
        if len(sign_reversals) >= 5:
            log("      ✓ 영역에 따라 피처 영향 방향이 반대")
            log("      → 단일 모델로는 이 패턴을 학습하기 어려움")
            log("      → Zone-specific Head 또는 Mixture-of-Experts 필요")
        else:
            log("      △ Sign Reversal 미미")

        # Signal 3: Magnitude Explosion (2x 이상 증폭)
        high_explosion = sum(1 for _, _, _, r in mag_changes if r >= 2.0)
        log(f"\n  [3] Magnitude Explosion (2x 이상 증폭): {high_explosion}개 피처")
        if high_explosion >= 5:
            log("      ✓ 경계 영역에서 특정 피처 영향력 폭증")
            log("      → 영역별 전문화된 Head 필요")
        else:
            log("      △ 증폭 현상 미미")

    log("\n" + "=" * 70)
    log("SHAP 분석 완료!")
    log("=" * 70)


if __name__ == '__main__':
    main()
