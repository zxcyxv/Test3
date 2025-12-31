# Feature Pipeline Overview

This document summarizes:
1) what `feature_engineering_v2.py` generates, and  
2) what additional feature transformations `main_train.py` applies before the model.

## 1) `feature_engineering_v2.py` Outputs

`feature_engineering_v2.py` augments `train_features_k8.csv` / `test_features_k8.csv` and produces
`train_features_v2.csv` / `test_features_v2.csv`.

It adds three groups of derived features:

### A) Geometric features (per timestep `i = 0..K-1`)
From `start_x_i`, `start_y_i`:
- `pressure_x_weight_i`: `(start_x_i / 105)^2`
- `is_zone14_i`: central zone flag (x in [70, 88.5] and y in [24, 44])
- `dist_to_goal_i`: distance to goal center (105, 34)
- `angle_visible_i`: `arctan(|34 - y| / (105 - x + eps))` (radians)
- `polar_angle_i`: `arctan2(34 - y, 105 - x)` (radians)
- `log_dist_to_goal_i`: `log(dist_to_goal_i + 1)`

### B) Sequence features (per timestep `i = 0..K-2`)
From `start_x_i, start_y_i, end_x_i, end_y_i`:
- `action_angle_i`: `arctan2(end_y_i - start_y_i, end_x_i - start_x_i)` (radians)
- `action_progress_i`: `end_x_i - start_x_i`
- `action_dist_i`: Euclidean move distance
- `action_lateral_i`: `end_y_i - start_y_i` (positive = up, negative = down)

Aggregates derived from recent actions (indices `K-2, K-3, K-4`):
- `prev_action_angle`, `prev_action_progress`, `prev_action_dist`, `prev_action_lateral`
- `recent_progress_sum`, `recent_progress_mean`, `recent_progress_std`
- `recent_dist_sum`, `recent_dist_mean`
- `recent_angle_std`
- `recent_lateral_sum`, `recent_lateral_abs_sum`

### C) Last-action features (single-row, derived from timestep `K-1`)
From `start_x_{K-1}, start_y_{K-1}`:
- `last_start_x`, `last_start_y`
- `last_pressure`, `last_dist_to_goal`, `last_angle_visible`
- `last_is_zone14`
- `last_polar_angle`, `last_log_dist_to_goal`
- `last_field_third` (x thirds: 0/1/2)
- `last_y_zone` (y thirds: 0/1/2)

## 2) `main_train.py` Input Transformations

`main_train.py` does **not** consume every engineered column. It uses a fixed set of base features
per timestep and applies additional preprocessing to make the model stable.

### A) Feature groups used by the model

**Coordinates (normalized by 105 for isotropy)**
- `start_x_t`, `start_y_t`, `end_x_t`, `end_y_t`

**Continuous (standardized using valid-only statistics)**
- `dt_t`, `ep_idx_norm_t`, `dist_to_goal_t`, `pressure_x_weight_t`
- `dx_t`, `dy_t`, `dist_t`, `speed_t`
- `action_progress_t`, `action_dist_t`, `action_lateral_t`

**Angles (converted to sin/cos, no standard scaling)**
- `angle_to_goal_t` (degrees in v1; converted to radians in `main_train.py`)
- `action_angle_t` (radians)
- `angle_visible_t` (radians)

**Categorical (offset by +1; 0 reserved for padding)**
- `type_id_t`, `res_id_t`, `is_home_t`, `x_zone_t`, `lane_t`, `is_zone14_t`

Note: `polar_angle_*`, `log_dist_to_goal_*`, `prev_*`, `recent_*`, `last_*` are **not** used
by `main_train.py` at the moment.

### B) Padding + valid masks

The dataset contains missing steps, especially at early timesteps (e.g., `t=0`).  
`main_train.py` treats these as padding:

- `valid_mask_seq` is created by checking `start_x_t` nullness.
- `padding_mask = ~valid_mask_seq` is passed into `MultiheadAttention` as `key_padding_mask`
  (so the transformer ignores padded steps).

### C) Angle encoding

Angles are encoded as `[sin(theta), cos(theta)]`:
- `angle_to_goal` is converted **deg → rad** before sin/cos.
- `action_angle`, `angle_visible` are already radians.

Y-flip augmentation flips the sine component for `angle_to_goal` and `action_angle`
(cosine stays unchanged).

### D) Scaling

**Coordinates**
- `start_*` / `end_*` are divided by **105.0** in `prepare_sequence_data`
  (keeps isotropic distance).

**Continuous**
- `cont_mean/cont_std` are computed using **only valid (non-padding) values**.
- Padding positions are reset to 0 after standardization.

### E) Categorical offset + embedding padding

Categorical IDs are shifted by +1:
- `0` is reserved for padding.
- Embeddings are created with `padding_idx=0` so padded positions do not contribute.

## Summary

`feature_engineering_v2.py` expands the raw data with geometric, sequential, and last-action
features. `main_train.py` then selects a specific subset, performs isotropic coordinate normalization,
valid-only continuous scaling, sin/cos angle encoding, and padding-aware attention.

This keeps the data physically consistent (angles, distances), avoids padding contamination,
and prevents categorical 0 from colliding with padding.
