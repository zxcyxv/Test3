"""
Stacking Model: Split model predictions as features for Unified model
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
        'early_stopping_rounds': 20,
        'verbosity': 0,
    }

    model = xgb.XGBRegressor(**params)
    if X_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        # No early stopping
        params.pop('early_stopping_rounds')
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)

    log(f"  [{name}] done")
    return model


def get_base_features(df):
    exclude_cols = ['game_episode', 'target_end_x', 'target_end_y', 'last_result_name', 'last_result_encoded']
    return [c for c in df.columns if c not in exclude_cols]


def main():
    log("Loading data...")
    df_success = pd.read_csv(DATA_DIR / 'train_features_k8_success.csv')
    df_fail = pd.read_csv(DATA_DIR / 'train_features_k8_fail.csv')
    df_all = pd.read_csv(DATA_DIR / 'train_features_k8.csv')
    test_df = pd.read_csv(DATA_DIR / 'test_features_k8.csv')

    base_features = get_base_features(df_all)
    log(f"  Success: {len(df_success)}, Fail: {len(df_fail)}, Test: {len(test_df)}")
    log(f"  Base features: {len(base_features)}")
    log("")

    # ===========================================
    # Step 1: Generate OOF predictions from split models
    # ===========================================
    log("Step 1: Generating OOF predictions from split models...")

    # Success OOF
    X_s = df_success[base_features].fillna(0)
    y_x_s = df_success['target_end_x'].values
    y_y_s = df_success['target_end_y'].values

    oof_pred_x_s = np.zeros(len(df_success))
    oof_pred_y_s = np.zeros(len(df_success))

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_s)):
        model_x = train_xgb(X_s.iloc[tr_idx], y_x_s[tr_idx], X_s.iloc[val_idx], y_x_s[val_idx], f"S_X_f{fold}")
        model_y = train_xgb(X_s.iloc[tr_idx], y_y_s[tr_idx], X_s.iloc[val_idx], y_y_s[val_idx], f"S_Y_f{fold}")
        oof_pred_x_s[val_idx] = model_x.predict(X_s.iloc[val_idx])
        oof_pred_y_s[val_idx] = model_y.predict(X_s.iloc[val_idx])

    log(f"  Success OOF Dist: {euclidean_distance(y_x_s, y_y_s, oof_pred_x_s, oof_pred_y_s):.3f}")

    # Fail OOF
    X_f = df_fail[base_features].fillna(0)
    y_x_f = df_fail['target_end_x'].values
    y_y_f = df_fail['target_end_y'].values

    oof_pred_x_f = np.zeros(len(df_fail))
    oof_pred_y_f = np.zeros(len(df_fail))

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_f)):
        model_x = train_xgb(X_f.iloc[tr_idx], y_x_f[tr_idx], X_f.iloc[val_idx], y_x_f[val_idx], f"F_X_f{fold}")
        model_y = train_xgb(X_f.iloc[tr_idx], y_y_f[tr_idx], X_f.iloc[val_idx], y_y_f[val_idx], f"F_Y_f{fold}")
        oof_pred_x_f[val_idx] = model_x.predict(X_f.iloc[val_idx])
        oof_pred_y_f[val_idx] = model_y.predict(X_f.iloc[val_idx])

    log(f"  Fail OOF Dist: {euclidean_distance(y_x_f, y_y_f, oof_pred_x_f, oof_pred_y_f):.3f}")
    log("")

    # ===========================================
    # Step 2: Train full split models for test prediction
    # ===========================================
    log("Step 2: Training full split models for test...")

    full_model_s_x = train_xgb(X_s, y_x_s, name="Full_S_X")
    full_model_s_y = train_xgb(X_s, y_y_s, name="Full_S_Y")
    full_model_f_x = train_xgb(X_f, y_x_f, name="Full_F_X")
    full_model_f_y = train_xgb(X_f, y_y_f, name="Full_F_Y")
    log("")

    # ===========================================
    # Step 3: Add split predictions as features
    # ===========================================
    log("Step 3: Adding split predictions as features...")

    # For training data: use OOF predictions
    df_all = df_all.copy()
    df_all['split_pred_x'] = 0.0
    df_all['split_pred_y'] = 0.0

    # Match by game_episode
    success_idx = df_all['last_result_name'] == 'Successful'
    fail_idx = df_all['last_result_name'] != 'Successful'

    # Map OOF predictions back
    success_map_x = dict(zip(df_success['game_episode'], oof_pred_x_s))
    success_map_y = dict(zip(df_success['game_episode'], oof_pred_y_s))
    fail_map_x = dict(zip(df_fail['game_episode'], oof_pred_x_f))
    fail_map_y = dict(zip(df_fail['game_episode'], oof_pred_y_f))

    df_all.loc[success_idx, 'split_pred_x'] = df_all.loc[success_idx, 'game_episode'].map(success_map_x)
    df_all.loc[success_idx, 'split_pred_y'] = df_all.loc[success_idx, 'game_episode'].map(success_map_y)
    df_all.loc[fail_idx, 'split_pred_x'] = df_all.loc[fail_idx, 'game_episode'].map(fail_map_x)
    df_all.loc[fail_idx, 'split_pred_y'] = df_all.loc[fail_idx, 'game_episode'].map(fail_map_y)

    # For test data: use full model predictions
    test_df = test_df.copy()
    X_test = test_df[base_features].fillna(0)
    test_success_idx = test_df['last_result_name'] == 'Successful'
    test_fail_idx = test_df['last_result_name'] != 'Successful'

    test_df['split_pred_x'] = 0.0
    test_df['split_pred_y'] = 0.0

    if test_success_idx.any():
        test_df.loc[test_success_idx, 'split_pred_x'] = full_model_s_x.predict(X_test[test_success_idx])
        test_df.loc[test_success_idx, 'split_pred_y'] = full_model_s_y.predict(X_test[test_success_idx])
    if test_fail_idx.any():
        test_df.loc[test_fail_idx, 'split_pred_x'] = full_model_f_x.predict(X_test[test_fail_idx])
        test_df.loc[test_fail_idx, 'split_pred_y'] = full_model_f_y.predict(X_test[test_fail_idx])

    log(f"  Added split_pred_x, split_pred_y features")
    log("")

    # ===========================================
    # Step 4: Train stacking model
    # ===========================================
    log("Step 4: Training stacking model...")

    stacking_features = base_features + ['last_result_encoded', 'split_pred_x', 'split_pred_y']

    X = df_all[stacking_features].fillna(0)
    y_x = df_all['target_end_x'].values
    y_y = df_all['target_end_y'].values

    X_tr, X_val, y_x_tr, y_x_val, y_y_tr, y_y_val = train_test_split(
        X, y_x, y_y, test_size=0.2, random_state=42
    )

    log(f"  Features: {len(stacking_features)}")
    log(f"  train={len(X_tr)}, val={len(X_val)}")

    stack_model_x = train_xgb(X_tr, y_x_tr, X_val, y_x_val, "Stack_X")
    stack_model_y = train_xgb(X_tr, y_y_tr, X_val, y_y_val, "Stack_Y")

    pred_x = np.clip(stack_model_x.predict(X_val), 0, FIELD_X)
    pred_y = np.clip(stack_model_y.predict(X_val), 0, FIELD_Y)
    dist = euclidean_distance(y_x_val, y_y_val, pred_x, pred_y)
    log(f"  Stacking Val Dist: {dist:.3f}")
    log("")

    # ===========================================
    # Step 5: Predict test
    # ===========================================
    log("Step 5: Predicting test...")

    X_test_stack = test_df[stacking_features].fillna(0)
    pred_test_x = np.clip(stack_model_x.predict(X_test_stack), 0, FIELD_X)
    pred_test_y = np.clip(stack_model_y.predict(X_test_stack), 0, FIELD_Y)

    submission = pd.DataFrame({
        'game_episode': test_df['game_episode'],
        'end_x': pred_test_x,
        'end_y': pred_test_y
    })
    output_path = DATA_DIR / 'submission_stacking_model.csv'
    submission.to_csv(output_path, index=False)
    log(f"\nSubmission saved: {output_path}")
    log(submission.head().to_string())


if __name__ == '__main__':
    main()
