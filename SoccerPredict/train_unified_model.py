"""
Unified Model: Single model with result as feature
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


def euclidean_distance(y_true_x, y_true_y, y_pred_x, y_pred_y):
    return np.mean(np.sqrt((y_true_x - y_pred_x)**2 + (y_true_y - y_pred_y)**2))


def get_feature_columns(df):
    exclude_cols = ['game_episode', 'target_end_x', 'target_end_y', 'last_result_name']
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


def main():
    log("Loading data...")

    # Load all data (not split)
    df_train = pd.read_csv(DATA_DIR / 'train_features_k5.csv')
    test_df = pd.read_csv(DATA_DIR / 'test_features_k5.csv')

    log(f"  Train: {len(df_train)}, Test: {len(test_df)}")

    # last_result_encoded is already a feature (0/1 for success/fail)
    X, y_x, y_y, feat_cols = prepare_data(df_train)

    X_tr, X_val, y_x_tr, y_x_val, y_y_tr, y_y_val = train_test_split(
        X, y_x, y_y, test_size=0.2, random_state=42
    )

    log(f"  train={len(X_tr)}, val={len(X_val)}")
    log(f"  Features: {len(feat_cols)}")
    log("")

    # Train unified models
    log("Training Unified models...")
    model_x = train_xgb_model(X_tr, y_x_tr, X_val, y_x_val, "Unified_X")
    model_y = train_xgb_model(X_tr, y_y_tr, X_val, y_y_val, "Unified_Y")

    # Validate
    pred_x = np.clip(model_x.predict(X_val), 0, FIELD_X)
    pred_y = np.clip(model_y.predict(X_val), 0, FIELD_Y)
    dist = euclidean_distance(y_x_val, y_y_val, pred_x, pred_y)
    log(f"  Unified Val Dist: {dist:.3f}")

    # Predict test
    log("\nPredicting test data...")
    X_test = test_df[feat_cols].fillna(0)
    pred_test_x = np.clip(model_x.predict(X_test), 0, FIELD_X)
    pred_test_y = np.clip(model_y.predict(X_test), 0, FIELD_Y)
    log("  Done!")

    # Submission
    submission = pd.DataFrame({
        'game_episode': test_df['game_episode'],
        'end_x': pred_test_x,
        'end_y': pred_test_y
    })
    output_path = DATA_DIR / 'submission_unified_model.csv'
    submission.to_csv(output_path, index=False)
    log(f"\nSubmission saved: {output_path}")
    log(submission.head().to_string())


if __name__ == '__main__':
    main()
