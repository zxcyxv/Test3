"""
Split Model Training: XGBoost GPU
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

DATA_DIR = Path('/workspace/SoccerPredict/open_track1')
MODEL_DIR = Path('/workspace/SoccerPredict/models')
MODEL_DIR.mkdir(exist_ok=True)

FIELD_X = 105
FIELD_Y = 68


def euclidean_distance(y_true_x, y_true_y, y_pred_x, y_pred_y):
    return np.mean(np.sqrt((y_true_x - y_pred_x)**2 + (y_true_y - y_pred_y)**2))


def get_feature_columns(df):
    # Split 모델에서는 last_result_encoded도 제외 (이미 분리되어 의미없음)
    exclude_cols = ['game_episode', 'target_end_x', 'target_end_y', 'last_result_name', 'last_result_encoded']
    return [c for c in df.columns if c not in exclude_cols]


def prepare_data(df):
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].fillna(0)
    y_x = df['target_end_x'].values
    y_y = df['target_end_y'].values
    return X, y_x, y_y, feature_cols


def train_xgb_model(X_train, y_train, X_val, y_val, name="model"):
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
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    log(f"  [{name}] done - best_iter: {model.best_iteration}")
    return model


def train_and_validate(df_success, df_fail):
    X_s, y_x_s, y_y_s, feat_cols = prepare_data(df_success)
    X_s_tr, X_s_val, y_x_s_tr, y_x_s_val, y_y_s_tr, y_y_s_val = train_test_split(
        X_s, y_x_s, y_y_s, test_size=0.2, random_state=42
    )

    X_f, y_x_f, y_y_f, _ = prepare_data(df_fail)
    X_f_tr, X_f_val, y_x_f_tr, y_x_f_val, y_y_f_tr, y_y_f_val = train_test_split(
        X_f, y_x_f, y_y_f, test_size=0.2, random_state=42
    )

    log(f"Success: train={len(X_s_tr)}, val={len(X_s_val)}")
    log(f"Fail: train={len(X_f_tr)}, val={len(X_f_val)}")
    log(f"Features: {len(feat_cols)}")
    log("")

    log("Training Success models...")
    model_s_x = train_xgb_model(X_s_tr, y_x_s_tr, X_s_val, y_x_s_val, "Success_X")
    model_s_y = train_xgb_model(X_s_tr, y_y_s_tr, X_s_val, y_y_s_val, "Success_Y")

    pred_s_x = np.clip(model_s_x.predict(X_s_val), 0, FIELD_X)
    pred_s_y = np.clip(model_s_y.predict(X_s_val), 0, FIELD_Y)
    dist_s = euclidean_distance(y_x_s_val, y_y_s_val, pred_s_x, pred_s_y)
    log(f"  Success Val Dist: {dist_s:.3f}")
    log("")

    log("Training Fail models...")
    model_f_x = train_xgb_model(X_f_tr, y_x_f_tr, X_f_val, y_x_f_val, "Fail_X")
    model_f_y = train_xgb_model(X_f_tr, y_y_f_tr, X_f_val, y_y_f_val, "Fail_Y")

    pred_f_x = np.clip(model_f_x.predict(X_f_val), 0, FIELD_X)
    pred_f_y = np.clip(model_f_y.predict(X_f_val), 0, FIELD_Y)
    dist_f = euclidean_distance(y_x_f_val, y_y_f_val, pred_f_x, pred_f_y)
    log(f"  Fail Val Dist: {dist_f:.3f}")
    log("")

    n_s, n_f = len(df_success), len(df_fail)
    weighted_dist = (dist_s * n_s + dist_f * n_f) / (n_s + n_f)
    log(f"Weighted Val Dist: {weighted_dist:.3f}")

    return {
        'success_x': model_s_x, 'success_y': model_s_y,
        'fail_x': model_f_x, 'fail_y': model_f_y,
        'feature_cols': feat_cols,
    }


def predict_test(models, test_df):
    log("Predicting test data...")

    feature_cols = models['feature_cols']
    X_test = test_df[feature_cols].fillna(0)

    is_success = test_df['last_result_name'] == 'Successful'
    log(f"  Success: {is_success.sum()}, Fail: {(~is_success).sum()}")

    pred_x = np.zeros(len(X_test))
    pred_y = np.zeros(len(X_test))

    if is_success.any():
        pred_x[is_success] = models['success_x'].predict(X_test[is_success])
        pred_y[is_success] = models['success_y'].predict(X_test[is_success])

    if (~is_success).any():
        pred_x[~is_success] = models['fail_x'].predict(X_test[~is_success])
        pred_y[~is_success] = models['fail_y'].predict(X_test[~is_success])

    pred_x = np.clip(pred_x, 0, FIELD_X)
    pred_y = np.clip(pred_y, 0, FIELD_Y)

    log("  Done!")
    return pred_x, pred_y


def main():
    log("Loading data...")

    try:
        df_success = pd.read_csv(DATA_DIR / 'train_features_k5_success.csv')
        df_fail = pd.read_csv(DATA_DIR / 'train_features_k5_fail.csv')
        test_df = pd.read_csv(DATA_DIR / 'test_features_k5.csv')
    except FileNotFoundError:
        log("ERROR: Run feature_engineering_k5.py first!")
        return None, None

    log(f"  Success: {len(df_success)}, Fail: {len(df_fail)}, Test: {len(test_df)}")
    log("")

    models = train_and_validate(df_success, df_fail)

    log("")
    pred_x, pred_y = predict_test(models, test_df)

    submission = pd.DataFrame({
        'game_episode': test_df['game_episode'],
        'end_x': pred_x,
        'end_y': pred_y
    })
    output_path = DATA_DIR / 'submission_split_model.csv'
    submission.to_csv(output_path, index=False)
    log(f"\nSubmission saved: {output_path}")
    log(submission.head().to_string())

    return models, submission


if __name__ == '__main__':
    main()
