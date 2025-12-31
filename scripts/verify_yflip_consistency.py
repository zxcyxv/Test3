import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from feature_engineering_v2 import FIELD_X, FIELD_Y, K, process_dataset


MODEL_BASES = [
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "dt",
    "ep_idx_norm",
    "x_zone",
    "lane",
    "type_id",
    "res_id",
    "is_home",
    "pressure_x_weight",
    "is_zone14",
    "angle_visible",
    "dx",
    "dy",
    "dist",
    "speed",
    "action_angle",
    "action_progress",
    "action_dist",
    "action_lateral",
    "dist_to_goal",
    "angle_to_goal",
]


def flip_raw_k8(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for t in range(K):
        sy = f"start_y_{t}"
        ey = f"end_y_{t}"
        if sy in df.columns:
            df[sy] = FIELD_Y - df[sy]
        if ey in df.columns:
            df[ey] = FIELD_Y - df[ey]

        dy = f"dy_{t}"
        if dy in df.columns:
            df[dy] = -df[dy]

        lat = f"action_lateral_{t}"
        if lat in df.columns:
            df[lat] = -df[lat]

        ang = f"action_angle_{t}"
        if ang in df.columns:
            df[ang] = -df[ang]

        ang_goal = f"angle_to_goal_{t}"
        if ang_goal in df.columns:
            df[ang_goal] = -df[ang_goal]

        lane = f"lane_{t}"
        if lane in df.columns:
            vals = df[lane].copy()
            df[lane] = np.where(vals == 0, 2, np.where(vals == 2, 0, vals))

    return df


def flip_v2_like_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for t in range(K):
        sy = f"start_y_{t}"
        ey = f"end_y_{t}"
        if sy in df.columns:
            df[sy] = FIELD_Y - df[sy]
        if ey in df.columns:
            df[ey] = FIELD_Y - df[ey]

        dy = f"dy_{t}"
        if dy in df.columns:
            df[dy] = -df[dy]

        lat = f"action_lateral_{t}"
        if lat in df.columns:
            df[lat] = -df[lat]

        ang_goal = f"angle_to_goal_{t}"
        if ang_goal in df.columns:
            df[ang_goal] = -df[ang_goal]

        ang = f"action_angle_{t}"
        if ang in df.columns:
            df[ang] = -df[ang]

        lane = f"lane_{t}"
        if lane in df.columns:
            vals = df[lane].copy()
            df[lane] = np.where(vals == 0, 2, np.where(vals == 2, 0, vals))

    return df


def main():
    path = "/workspace/SoccerPredict/open_track1/train_features_k8.csv"
    df_raw = pd.read_csv(path)

    sample = df_raw.sample(n=min(len(df_raw), 2000), random_state=42).reset_index(drop=True)
    v2 = process_dataset(sample, name="Sample")
    v2_flip_raw = process_dataset(flip_raw_k8(sample), name="Sample(Y-flip)")

    v2_flip_logic = flip_v2_like_pipeline(v2)

    cols = []
    for base in MODEL_BASES:
        for t in range(K):
            col = f"{base}_{t}"
            if col in v2.columns:
                cols.append(col)

    mismatches = []
    angle_cols = [c for c in cols if c.startswith("action_angle_") or c.startswith("angle_to_goal_")]
    for col in cols:
        a = v2_flip_logic[col].values
        b = v2_flip_raw[col].values
        if a.dtype.kind in "if":
            if col in angle_cols:
                # Compare angles modulo 2*pi
                diff = np.nanmax(np.abs(np.arctan2(np.sin(a - b), np.cos(a - b))))
                if diff > 1e-6:
                    mismatches.append((col, float(diff)))
            else:
                diff = np.nanmax(np.abs(a - b))
                if diff > 1e-6:
                    mismatches.append((col, float(diff)))
        else:
            neq = (a != b).sum()
            if neq > 0:
                mismatches.append((col, int(neq)))

    print("Checked columns:", len(cols))
    if not mismatches:
        print("OK: flip logic matches FE on all checked columns.")
    else:
        print("MISMATCHES:")
        for item in mismatches[:50]:
            print(item)
        if len(mismatches) > 50:
            print(f"... and {len(mismatches) - 50} more")


if __name__ == "__main__":
    main()
