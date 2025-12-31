import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split


DATA_PATH = "/workspace/SoccerPredict/open_track1/xgb_hybrid_features.csv"


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X = df.drop(["target_end_x", "target_end_y"], axis=1)
    y_x = df["target_end_x"].values
    y_y = df["target_end_y"].values

    X_train, X_val, yx_train, yx_val, yy_train, yy_val = train_test_split(
        X, y_x, y_y, test_size=0.2, random_state=42
    )

    params = {
        "n_estimators": 1000,
        "max_depth": 7,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "device": "cuda",
        "objective": "reg:absoluteerror",
    }

    print("Training XGBoost for X-coordinate...")
    model_x = xgb.XGBRegressor(**params)
    params["early_stopping_rounds"] = 50
    model_x.fit(
        X_train,
        yx_train,
        eval_set=[(X_val, yx_val)],
        verbose=100,
    )

    print("\nTraining XGBoost for Y-coordinate...")
    model_y = xgb.XGBRegressor(**params)
    model_y.fit(
        X_train,
        yy_train,
        eval_set=[(X_val, yy_val)],
        verbose=100,
    )

    pred_x = model_x.predict(X_val)
    pred_y = model_y.predict(X_val)
    dist = np.sqrt((pred_x - yx_val) ** 2 + (pred_y - yy_val) ** 2).mean()
    print(f"\n[Final Hybrid Result] Validation Distance: {dist:.4f}m")


if __name__ == "__main__":
    main()
