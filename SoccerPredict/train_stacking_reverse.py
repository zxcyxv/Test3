"""
Reverse Stacking: Unified model predictions as features for Split models
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


def train_xgb(X_train, y_train, X_val=None, y_val=None, name="model"):
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

    log(f"  [{name}] done")
    return model


def get_base_features(df):
    exclude_cols = ['game_episode', 'target_end_x', 'target_end_y', 'last_result_name']
    return [c for c in df.columns if c not in exclude_cols]


def main():
    log("Loading data...")
    df_all = pd.read_csv(DATA_DIR / 'train_features_k5.csv')
    test_df = pd.read_csv(DATA_DIR / 'test_features_k5.csv')

    base_features = get_base_features(df_all)  # includes last_result_encoded
    log(f"  Train: {len(df_all)}, Test: {len(test_df)}")
    log(f"  Base features: {len(base_features)}")
    log("")

    # ===========================================
    # Step 1: Generate OOF predictions from Unified model
    # ===========================================
    log("Step 1: Generating OOF predictions from Unified model...")

    X_all = df_all[base_features].fillna(0)
    y_x_all = df_all['target_end_x'].values
    y_y_all = df_all['target_end_y'].values

    oof_pred_x = np.zeros(len(df_all))
    oof_pred_y = np.zeros(len(df_all))

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_all)):
        model_x = train_xgb(X_all.iloc[tr_idx], y_x_all[tr_idx],
                           X_all.iloc[val_idx], y_x_all[val_idx], f"U_X_f{fold}")
        model_y = train_xgb(X_all.iloc[tr_idx], y_y_all[tr_idx],
                           X_all.iloc[val_idx], y_y_all[val_idx], f"U_Y_f{fold}")
        oof_pred_x[val_idx] = model_x.predict(X_all.iloc[val_idx])
        oof_pred_y[val_idx] = model_y.predict(X_all.iloc[val_idx])

    log(f"  Unified OOF Dist: {euclidean_distance(y_x_all, y_y_all, oof_pred_x, oof_pred_y):.3f}")
    log("")

    # ===========================================
    # Step 2: Train full Unified model for test prediction
    # ===========================================
    log("Step 2: Training full Unified model for test...")
    full_model_x = train_xgb(X_all, y_x_all, name="Full_U_X")
    full_model_y = train_xgb(X_all, y_y_all, name="Full_U_Y")

    X_test = test_df[base_features].fillna(0)
    test_unified_pred_x = full_model_x.predict(X_test)
    test_unified_pred_y = full_model_y.predict(X_test)
    log("")

    # ===========================================
    # Step 3: Add Unified predictions as features to train/test
    # ===========================================
    log("Step 3: Adding Unified predictions as features...")

    df_all['unified_pred_x'] = oof_pred_x
    df_all['unified_pred_y'] = oof_pred_y
    test_df['unified_pred_x'] = test_unified_pred_x
    test_df['unified_pred_y'] = test_unified_pred_y

    # Split features (without last_result_encoded, with unified preds)
    split_features = [f for f in base_features if f != 'last_result_encoded'] + ['unified_pred_x', 'unified_pred_y']
    log(f"  Split features: {len(split_features)}")
    log("")

    # ===========================================
    # Step 4: Train Split models with Unified predictions
    # ===========================================
    log("Step 4: Training Split models with Unified predictions...")

    # Split data
    success_mask = df_all['last_result_name'] == 'Successful'
    fail_mask = ~success_mask

    df_success = df_all[success_mask].copy()
    df_fail = df_all[fail_mask].copy()

    # Success model
    X_s = df_success[split_features].fillna(0)
    y_x_s = df_success['target_end_x'].values
    y_y_s = df_success['target_end_y'].values

    X_s_tr, X_s_val, y_x_s_tr, y_x_s_val, y_y_s_tr, y_y_s_val = train_test_split(
        X_s, y_x_s, y_y_s, test_size=0.2, random_state=42
    )

    log(f"  Success: train={len(X_s_tr)}, val={len(X_s_val)}")
    model_s_x = train_xgb(X_s_tr, y_x_s_tr, X_s_val, y_x_s_val, "S_X")
    model_s_y = train_xgb(X_s_tr, y_y_s_tr, X_s_val, y_y_s_val, "S_Y")

    pred_s_x = np.clip(model_s_x.predict(X_s_val), 0, FIELD_X)
    pred_s_y = np.clip(model_s_y.predict(X_s_val), 0, FIELD_Y)
    dist_s = euclidean_distance(y_x_s_val, y_y_s_val, pred_s_x, pred_s_y)
    log(f"  Success Val Dist: {dist_s:.3f}")

    # Fail model
    X_f = df_fail[split_features].fillna(0)
    y_x_f = df_fail['target_end_x'].values
    y_y_f = df_fail['target_end_y'].values

    X_f_tr, X_f_val, y_x_f_tr, y_x_f_val, y_y_f_tr, y_y_f_val = train_test_split(
        X_f, y_x_f, y_y_f, test_size=0.2, random_state=42
    )

    log(f"  Fail: train={len(X_f_tr)}, val={len(X_f_val)}")
    model_f_x = train_xgb(X_f_tr, y_x_f_tr, X_f_val, y_x_f_val, "F_X")
    model_f_y = train_xgb(X_f_tr, y_y_f_tr, X_f_val, y_y_f_val, "F_Y")

    pred_f_x = np.clip(model_f_x.predict(X_f_val), 0, FIELD_X)
    pred_f_y = np.clip(model_f_y.predict(X_f_val), 0, FIELD_Y)
    dist_f = euclidean_distance(y_x_f_val, y_y_f_val, pred_f_x, pred_f_y)
    log(f"  Fail Val Dist: {dist_f:.3f}")

    # Weighted
    n_s, n_f = len(df_success), len(df_fail)
    weighted_dist = (dist_s * n_s + dist_f * n_f) / (n_s + n_f)
    log(f"  Weighted Val Dist: {weighted_dist:.3f}")
    log("")

    # ===========================================
    # Step 5: Predict test
    # ===========================================
    log("Step 5: Predicting test...")

    test_success_mask = test_df['last_result_name'] == 'Successful'
    test_fail_mask = ~test_success_mask

    X_test_split = test_df[split_features].fillna(0)

    pred_test_x = np.zeros(len(test_df))
    pred_test_y = np.zeros(len(test_df))

    if test_success_mask.any():
        pred_test_x[test_success_mask] = model_s_x.predict(X_test_split[test_success_mask])
        pred_test_y[test_success_mask] = model_s_y.predict(X_test_split[test_success_mask])
    if test_fail_mask.any():
        pred_test_x[test_fail_mask] = model_f_x.predict(X_test_split[test_fail_mask])
        pred_test_y[test_fail_mask] = model_f_y.predict(X_test_split[test_fail_mask])

    pred_test_x = np.clip(pred_test_x, 0, FIELD_X)
    pred_test_y = np.clip(pred_test_y, 0, FIELD_Y)

    log(f"  Success: {test_success_mask.sum()}, Fail: {test_fail_mask.sum()}")

    submission = pd.DataFrame({
        'game_episode': test_df['game_episode'],
        'end_x': pred_test_x,
        'end_y': pred_test_y
    })
    output_path = DATA_DIR / 'submission_stacking_reverse.csv'
    submission.to_csv(output_path, index=False)
    log(f"\nSubmission saved: {output_path}")
    log(submission.head().to_string())


if __name__ == '__main__':
    main()
