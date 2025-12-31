"""
경계 분류기: 패스가 어디로 나가는지 분류
0: 필드 안 (In-field)
1: 위쪽 라인 아웃 (Y > 63)
2: 아래쪽 라인 아웃 (Y < 5)
3: 골라인 아웃 (X > 100 or X < 5)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

DATA_DIR = Path('/workspace/SoccerPredict/open_track1')
FIELD_X = 105
FIELD_Y = 68


def create_boundary_labels(end_x, end_y, y_threshold=5, x_threshold=5):
    """
    0: In-field
    1: Top out (Y > 68 - threshold)
    2: Bottom out (Y < threshold)
    3: Goal-line out (X > 105 - threshold or X < threshold)
    """
    labels = np.zeros(len(end_x), dtype=int)

    # 골라인 아웃 (우선순위 높음 - 코너 처리)
    labels[(end_x > FIELD_X - x_threshold) | (end_x < x_threshold)] = 3

    # Y 라인 아웃 (골라인이 아닌 경우만)
    labels[(labels == 0) & (end_y > FIELD_Y - y_threshold)] = 1  # Top
    labels[(labels == 0) & (end_y < y_threshold)] = 2  # Bottom

    return labels


def main():
    log("Loading data...")
    df_all = pd.read_csv(DATA_DIR / 'train_features_k8.csv')

    exclude = ['game_episode', 'target_end_x', 'target_end_y', 'last_result_name', 'last_result_encoded']
    base_features = [c for c in df_all.columns if c not in exclude]
    unified_features = base_features + ['last_result_encoded']

    # Train/Val split
    train_idx, val_idx = train_test_split(range(len(df_all)), test_size=0.2, random_state=42)
    df_train = df_all.iloc[train_idx].copy().reset_index(drop=True)
    df_val = df_all.iloc[val_idx].copy().reset_index(drop=True)

    log(f"Train: {len(df_train)}, Val: {len(df_val)}")

    # Create boundary labels
    log("\n" + "=" * 60)
    log("경계 라벨 생성 (threshold=5)")
    log("=" * 60)

    train_labels = create_boundary_labels(
        df_train['target_end_x'].values,
        df_train['target_end_y'].values
    )
    val_labels = create_boundary_labels(
        df_val['target_end_x'].values,
        df_val['target_end_y'].values
    )

    label_names = ['In-field', 'Top-out', 'Bottom-out', 'Goal-line']

    log("\n[Train 라벨 분포]")
    for i, name in enumerate(label_names):
        count = (train_labels == i).sum()
        pct = count / len(train_labels) * 100
        log(f"  {i} ({name}): {count} ({pct:.1f}%)")

    log("\n[Val 라벨 분포]")
    for i, name in enumerate(label_names):
        count = (val_labels == i).sum()
        pct = count / len(val_labels) * 100
        log(f"  {i} ({name}): {count} ({pct:.1f}%)")

    # ============================================
    # 분류기 학습
    # ============================================
    log("\n" + "=" * 60)
    log("분류기 학습")
    log("=" * 60)

    X_train = df_train[unified_features].fillna(0)
    X_val = df_val[unified_features].fillna(0)

    clf = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=4,
        tree_method='hist',
        device='cuda',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=200,
        verbosity=0,
    )

    clf.fit(X_train, train_labels)
    log("분류기 학습 완료!")

    # 예측
    pred_labels = clf.predict(X_val)
    pred_proba = clf.predict_proba(X_val)

    # ============================================
    # 분류 성능
    # ============================================
    log("\n" + "=" * 60)
    log("분류 성능")
    log("=" * 60)

    log("\n[Classification Report]")
    print(classification_report(val_labels, pred_labels, target_names=label_names))

    log("\n[Confusion Matrix]")
    cm = confusion_matrix(val_labels, pred_labels)
    log(f"         Pred: {label_names}")
    for i, name in enumerate(label_names):
        log(f"  True {name:12s}: {cm[i]}")

    # ============================================
    # Success vs Fail 별 분류 성능
    # ============================================
    log("\n" + "=" * 60)
    log("Success vs Fail 별 분류 성능")
    log("=" * 60)

    is_success = df_val['last_result_name'] == 'Successful'

    for name, mask in [("Success", is_success), ("Fail", ~is_success)]:
        log(f"\n[{name} 샘플]")
        log(f"  샘플 수: {mask.sum()}")

        # 실제 분포
        log(f"  실제 분포:")
        for i, lname in enumerate(label_names):
            count = (val_labels[mask] == i).sum()
            pct = count / mask.sum() * 100 if mask.sum() > 0 else 0
            log(f"    {lname}: {count} ({pct:.1f}%)")

        # 정확도
        acc = (pred_labels[mask] == val_labels[mask]).mean() * 100
        log(f"  분류 정확도: {acc:.1f}%")

    # ============================================
    # 경계 예측 재현율 (Recall)
    # ============================================
    log("\n" + "=" * 60)
    log("경계 예측 Recall (실제 아웃일 때 아웃으로 예측)")
    log("=" * 60)

    for i, name in enumerate(label_names):
        if i == 0:
            continue  # In-field 스킵

        true_mask = val_labels == i
        if true_mask.sum() > 0:
            pred_mask = pred_labels == i
            recall = (true_mask & pred_mask).sum() / true_mask.sum() * 100
            precision = (true_mask & pred_mask).sum() / pred_mask.sum() * 100 if pred_mask.sum() > 0 else 0
            log(f"  {name}: Recall={recall:.1f}%, Precision={precision:.1f}% (n={true_mask.sum()})")

    # ============================================
    # 분류기 활용 회귀 전략
    # ============================================
    log("\n" + "=" * 60)
    log("분류기 활용 회귀 전략 테스트")
    log("=" * 60)

    # 회귀 모델 학습
    log("\n회귀 모델 학습 중...")
    model_x = xgb.XGBRegressor(
        objective='reg:squarederror', tree_method='hist', device='cuda',
        max_depth=6, learning_rate=0.1, n_estimators=200, verbosity=0
    )
    model_y = xgb.XGBRegressor(
        objective='reg:squarederror', tree_method='hist', device='cuda',
        max_depth=6, learning_rate=0.1, n_estimators=200, verbosity=0
    )

    model_x.fit(X_train, df_train['target_end_x'].values)
    model_y.fit(X_train, df_train['target_end_y'].values)

    pred_x = model_x.predict(X_val)
    pred_y = model_y.predict(X_val)

    true_x = df_val['target_end_x'].values
    true_y = df_val['target_end_y'].values

    # 베이스라인
    base_pred_x = np.clip(pred_x, 0, FIELD_X)
    base_pred_y = np.clip(pred_y, 0, FIELD_Y)
    base_dist = np.sqrt((true_x - base_pred_x)**2 + (true_y - base_pred_y)**2).mean()
    log(f"\n베이스라인 (Unified): {base_dist:.4f}")

    # 전략: 분류기 예측에 따라 후처리
    log("\n[전략: 분류기 예측 기반 후처리]")

    for conf_threshold in [0.3, 0.5, 0.7]:
        new_x = base_pred_x.copy()
        new_y = base_pred_y.copy()

        # 높은 확률로 아웃 예측된 경우 경계로 밀기
        for i in range(len(pred_proba)):
            max_prob = pred_proba[i].max()
            pred_class = pred_labels[i]

            if max_prob >= conf_threshold and pred_class != 0:
                if pred_class == 1:  # Top out
                    new_y[i] = max(new_y[i], FIELD_Y - 3)  # 65 이상으로
                elif pred_class == 2:  # Bottom out
                    new_y[i] = min(new_y[i], 3)  # 3 이하로
                elif pred_class == 3:  # Goal-line out
                    if new_x[i] > FIELD_X / 2:
                        new_x[i] = max(new_x[i], FIELD_X - 3)
                    else:
                        new_x[i] = min(new_x[i], 3)

        new_x = np.clip(new_x, 0, FIELD_X)
        new_y = np.clip(new_y, 0, FIELD_Y)

        dist = np.sqrt((true_x - new_x)**2 + (true_y - new_y)**2).mean()
        improvement = base_dist - dist
        log(f"  conf_threshold={conf_threshold}: {dist:.4f} (개선: {improvement:+.4f})")

    # 전략: 아웃 예측 시 경계로 강제
    log("\n[전략: 아웃 예측 시 경계로 강제]")

    for conf_threshold in [0.5, 0.6, 0.7, 0.8]:
        new_x = base_pred_x.copy()
        new_y = base_pred_y.copy()

        for i in range(len(pred_proba)):
            max_prob = pred_proba[i].max()
            pred_class = pred_labels[i]

            if max_prob >= conf_threshold and pred_class != 0:
                if pred_class == 1:  # Top out
                    new_y[i] = FIELD_Y
                elif pred_class == 2:  # Bottom out
                    new_y[i] = 0
                elif pred_class == 3:  # Goal-line out
                    if new_x[i] > FIELD_X / 2:
                        new_x[i] = FIELD_X
                    else:
                        new_x[i] = 0

        dist = np.sqrt((true_x - new_x)**2 + (true_y - new_y)**2).mean()
        improvement = base_dist - dist
        log(f"  conf_threshold={conf_threshold}: {dist:.4f} (개선: {improvement:+.4f})")

    # ============================================
    # Feature Importance
    # ============================================
    log("\n" + "=" * 60)
    log("분류기 Feature Importance (Top 15)")
    log("=" * 60)

    importance = clf.feature_importances_
    feature_importance = list(zip(unified_features, importance))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for feat, imp in feature_importance[:15]:
        log(f"  {feat}: {imp:.4f}")


if __name__ == '__main__':
    main()
