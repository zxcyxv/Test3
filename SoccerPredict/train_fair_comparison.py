"""
Fair Comparison with OOF for Stacking
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

DATA_DIR = Path('/workspace/SoccerPredict/open_track1')
FIELD_X = 105
FIELD_Y = 68


def euclidean_distance(y_true_x, y_true_y, y_pred_x, y_pred_y):
    return np.mean(np.sqrt((y_true_x - y_pred_x)**2 + (y_true_y - y_pred_y)**2))


def train_xgb(X_train, y_train, X_val=None, y_val=None, name="model", silent=False):
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
    if not silent:
        log(f"  [{name}] done")
    return model


def main():
    log("Loading data...")
    df_all = pd.read_csv(DATA_DIR / 'train_features_k5.csv')

    exclude = ['game_episode', 'target_end_x', 'target_end_y', 'last_result_name', 'last_result_encoded']
    base_features = [c for c in df_all.columns if c not in exclude]
    unified_features = base_features + ['last_result_encoded']

    log(f"  Total: {len(df_all)}")
    log(f"  Base features: {len(base_features)}, Unified features: {len(unified_features)}")

    # 동일한 train/val split
    train_idx, val_idx = train_test_split(range(len(df_all)), test_size=0.2, random_state=42)

    df_train = df_all.iloc[train_idx].copy().reset_index(drop=True)
    df_val = df_all.iloc[val_idx].copy().reset_index(drop=True)

    log(f"  Train: {len(df_train)}, Val: {len(df_val)}")
    log("")

    # ============================================
    # Model 1: Split (분리 모델)
    # ============================================
    log("=" * 50)
    log("Model 1: Split (Success/Fail 분리)")
    log("=" * 50)

    train_success = df_train[df_train['last_result_name'] == 'Successful']
    train_fail = df_train[df_train['last_result_name'] != 'Successful']
    val_success = df_val[df_val['last_result_name'] == 'Successful']
    val_fail = df_val[df_val['last_result_name'] != 'Successful']

    log(f"  Train - Success: {len(train_success)}, Fail: {len(train_fail)}")
    log(f"  Val - Success: {len(val_success)}, Fail: {len(val_fail)}")

    # Success model
    X_s_tr = train_success[base_features].fillna(0)
    X_s_val = val_success[base_features].fillna(0)

    model_s_x = train_xgb(X_s_tr, train_success['target_end_x'].values,
                          X_s_val, val_success['target_end_x'].values, "Split_S_X")
    model_s_y = train_xgb(X_s_tr, train_success['target_end_y'].values,
                          X_s_val, val_success['target_end_y'].values, "Split_S_Y")

    pred_s_x = np.clip(model_s_x.predict(X_s_val), 0, FIELD_X)
    pred_s_y = np.clip(model_s_y.predict(X_s_val), 0, FIELD_Y)
    dist_s = euclidean_distance(val_success['target_end_x'].values,
                                 val_success['target_end_y'].values, pred_s_x, pred_s_y)
    log(f"  Success Val Dist: {dist_s:.3f}")

    # Fail model
    X_f_tr = train_fail[base_features].fillna(0)
    X_f_val = val_fail[base_features].fillna(0)

    model_f_x = train_xgb(X_f_tr, train_fail['target_end_x'].values,
                          X_f_val, val_fail['target_end_x'].values, "Split_F_X")
    model_f_y = train_xgb(X_f_tr, train_fail['target_end_y'].values,
                          X_f_val, val_fail['target_end_y'].values, "Split_F_Y")

    pred_f_x = np.clip(model_f_x.predict(X_f_val), 0, FIELD_X)
    pred_f_y = np.clip(model_f_y.predict(X_f_val), 0, FIELD_Y)
    dist_f = euclidean_distance(val_fail['target_end_x'].values,
                                 val_fail['target_end_y'].values, pred_f_x, pred_f_y)
    log(f"  Fail Val Dist: {dist_f:.3f}")

    split_weighted = (dist_s * len(val_success) + dist_f * len(val_fail)) / len(df_val)
    log(f"  >> Split Weighted Dist: {split_weighted:.3f}")
    log("")

    # ============================================
    # Model 2: Unified (통합 모델)
    # ============================================
    log("=" * 50)
    log("Model 2: Unified (result 피처 포함)")
    log("=" * 50)

    X_tr = df_train[unified_features].fillna(0)
    X_val = df_val[unified_features].fillna(0)

    model_u_x = train_xgb(X_tr, df_train['target_end_x'].values,
                          X_val, df_val['target_end_x'].values, "Unified_X")
    model_u_y = train_xgb(X_tr, df_train['target_end_y'].values,
                          X_val, df_val['target_end_y'].values, "Unified_Y")

    pred_u_x = np.clip(model_u_x.predict(X_val), 0, FIELD_X)
    pred_u_y = np.clip(model_u_y.predict(X_val), 0, FIELD_Y)
    unified_dist = euclidean_distance(df_val['target_end_x'].values,
                                       df_val['target_end_y'].values, pred_u_x, pred_u_y)
    log(f"  >> Unified Dist: {unified_dist:.3f}")
    log("")

    # ============================================
    # Model 3: Stacking with OOF (Unified → Split)
    # ============================================
    log("=" * 50)
    log("Model 3: Stacking with OOF (Unified 예측 → Split 피처)")
    log("=" * 50)

    # Train 데이터에 대해 OOF 예측 생성
    log("  Generating OOF predictions for train...")
    oof_pred_x = np.zeros(len(df_train))
    oof_pred_y = np.zeros(len(df_train))

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr_idx, oof_idx) in enumerate(kf.split(df_train)):
        X_fold_tr = df_train.iloc[tr_idx][unified_features].fillna(0)
        y_fold_tr_x = df_train.iloc[tr_idx]['target_end_x'].values
        y_fold_tr_y = df_train.iloc[tr_idx]['target_end_y'].values

        X_fold_oof = df_train.iloc[oof_idx][unified_features].fillna(0)

        m_x = train_xgb(X_fold_tr, y_fold_tr_x, name=f"OOF_X_f{fold}", silent=True)
        m_y = train_xgb(X_fold_tr, y_fold_tr_y, name=f"OOF_Y_f{fold}", silent=True)

        oof_pred_x[oof_idx] = m_x.predict(X_fold_oof)
        oof_pred_y[oof_idx] = m_y.predict(X_fold_oof)

    log(f"  OOF predictions generated (5-fold)")

    # Val에 대해서는 전체 train으로 학습한 unified 모델 사용
    val_pred_x = model_u_x.predict(X_val)
    val_pred_y = model_u_y.predict(X_val)

    # 피처 추가
    df_train['unified_pred_x'] = oof_pred_x
    df_train['unified_pred_y'] = oof_pred_y
    df_val['unified_pred_x'] = val_pred_x
    df_val['unified_pred_y'] = val_pred_y

    stack_features = base_features + ['unified_pred_x', 'unified_pred_y']
    log(f"  Stack features: {len(stack_features)}")

    # Split with stacking
    train_success = df_train[df_train['last_result_name'] == 'Successful']
    train_fail = df_train[df_train['last_result_name'] != 'Successful']
    val_success = df_val[df_val['last_result_name'] == 'Successful']
    val_fail = df_val[df_val['last_result_name'] != 'Successful']

    # Success model
    X_s_tr = train_success[stack_features].fillna(0)
    X_s_val = val_success[stack_features].fillna(0)

    model_stack_s_x = train_xgb(X_s_tr, train_success['target_end_x'].values,
                                 X_s_val, val_success['target_end_x'].values, "Stack_S_X")
    model_stack_s_y = train_xgb(X_s_tr, train_success['target_end_y'].values,
                                 X_s_val, val_success['target_end_y'].values, "Stack_S_Y")

    pred_stack_s_x = np.clip(model_stack_s_x.predict(X_s_val), 0, FIELD_X)
    pred_stack_s_y = np.clip(model_stack_s_y.predict(X_s_val), 0, FIELD_Y)
    stack_dist_s = euclidean_distance(val_success['target_end_x'].values,
                                       val_success['target_end_y'].values,
                                       pred_stack_s_x, pred_stack_s_y)
    log(f"  Success Val Dist: {stack_dist_s:.3f}")

    # Fail model
    X_f_tr = train_fail[stack_features].fillna(0)
    X_f_val = val_fail[stack_features].fillna(0)

    model_stack_f_x = train_xgb(X_f_tr, train_fail['target_end_x'].values,
                                 X_f_val, val_fail['target_end_x'].values, "Stack_F_X")
    model_stack_f_y = train_xgb(X_f_tr, train_fail['target_end_y'].values,
                                 X_f_val, val_fail['target_end_y'].values, "Stack_F_Y")

    pred_stack_f_x = np.clip(model_stack_f_x.predict(X_f_val), 0, FIELD_X)
    pred_stack_f_y = np.clip(model_stack_f_y.predict(X_f_val), 0, FIELD_Y)
    stack_dist_f = euclidean_distance(val_fail['target_end_x'].values,
                                       val_fail['target_end_y'].values,
                                       pred_stack_f_x, pred_stack_f_y)
    log(f"  Fail Val Dist: {stack_dist_f:.3f}")

    stack_weighted = (stack_dist_s * len(val_success) + stack_dist_f * len(val_fail)) / len(df_val)
    log(f"  >> Stacking Weighted Dist: {stack_weighted:.3f}")
    log("")

    # ============================================
    # Summary
    # ============================================
    log("=" * 50)
    log("SUMMARY")
    log("=" * 50)
    log(f"  Split (분리):              {split_weighted:.3f}")
    log(f"  Unified (통합):            {unified_dist:.3f}")
    log(f"  Stacking OOF (Unified→Split): {stack_weighted:.3f}")


if __name__ == '__main__':
    main()
