import numpy as np

from .config import (
    ANGLE_FEATURES,
    ANGLE_FLIP_FEATURES,
    CAT_FEATURES,
    CONT_FEATURES,
    COORD_FEATURES,
    FIELD_X,
    FIELD_Y,
    K,
    MAX_LEN,
)


def create_boundary_labels(end_x, end_y):
    labels = np.zeros(len(end_x), dtype=int)
    labels[(end_x > FIELD_X - 5) | (end_x < 5)] = 3
    labels[(labels == 0) & (end_y > FIELD_Y - 5)] = 1
    labels[(labels == 0) & (end_y < 5)] = 2
    return labels


def compute_boundary_distance(end_x, end_y):
    dist_left = end_x
    dist_right = FIELD_X - end_x
    dist_bottom = end_y
    dist_top = FIELD_Y - end_y
    min_dist = np.minimum(np.minimum(dist_left, dist_right), np.minimum(dist_bottom, dist_top))
    return min_dist / (FIELD_X / 2)


def prepare_sequence_data(df, k: int = K):
    n_samples = len(df)
    coord_dim = 4
    cont_dim = len(CONT_FEATURES)
    angle_dim = len(ANGLE_FEATURES) * 2
    cat_dim = len(CAT_FEATURES)

    coords_seq = np.full((n_samples, k, coord_dim), -1.0)
    cont_seq = np.zeros((n_samples, k, cont_dim))
    angle_seq = np.zeros((n_samples, k, angle_dim))
    cat_seq = np.zeros((n_samples, k, cat_dim), dtype=int)
    valid_mask_seq = np.zeros((n_samples, k), dtype=bool)
    cont_valid_mask = np.zeros((n_samples, k, cont_dim), dtype=bool)

    coord_idx = {"start_x": 0, "start_y": 1, "end_x": 2, "end_y": 3}
    cont_idx = {feat: i for i, feat in enumerate(CONT_FEATURES)}
    angle_idx = {feat: i for i, feat in enumerate(ANGLE_FEATURES)}
    cat_idx = {feat: i for i, feat in enumerate(CAT_FEATURES)}

    for t in range(k):
        check_col = f"start_x_{t}"
        if check_col not in df.columns:
            continue

        has_data = ~df[check_col].isna()
        valid_mask_seq[has_data, t] = True

        for feat in COORD_FEATURES:
            col = f"{feat}_{t}"
            if col in df.columns and (feat in COORD_FEATURES[:2] or t < k - 1):
                vals = df.loc[has_data, col].values / MAX_LEN
                coords_seq[has_data, t, coord_idx[feat]] = vals

        for feat in CONT_FEATURES:
            col = f"{feat}_{t}"
            if col in df.columns and (feat in CONT_FEATURES[:4] or t < k - 1):
                cont_seq[has_data, t, cont_idx[feat]] = df.loc[has_data, col].values
                cont_valid_mask[has_data, t, cont_idx[feat]] = True

        for feat in ANGLE_FEATURES:
            col = f"{feat}_{t}"
            if col in df.columns and (feat in ("angle_to_goal", "angle_visible") or t < k - 1):
                rads = df.loc[has_data, col].values
                if feat == "angle_to_goal":
                    rads = np.deg2rad(rads)
                idx = angle_idx[feat]
                angle_seq[has_data, t, idx * 2] = np.sin(rads)
                angle_seq[has_data, t, idx * 2 + 1] = np.cos(rads)

        for feat in CAT_FEATURES:
            col = f"{feat}_{t}"
            if col in df.columns:
                cat_seq[has_data, t, cat_idx[feat]] = df.loc[has_data, col].astype(int).values + 1

    return (
        coords_seq,
        cont_seq,
        angle_seq,
        cat_seq,
        valid_mask_seq,
        cont_valid_mask,
        cont_idx,
        angle_idx,
        cat_idx,
    )


def augment_data(
    coords,
    cont,
    angles,
    cat,
    padding_mask,
    cont_valid_mask,
    y_coords,
    boundary_dist,
    boundary_zone,
    cont_idx,
    angle_idx,
    cat_idx,
):
    coords_all = [coords.copy()]
    cont_all = [cont.copy()]
    angles_all = [angles.copy()]
    cat_all = [cat.copy()]
    mask_all = [padding_mask.copy()]
    cont_valid_all = [cont_valid_mask.copy()]
    y_coords_all = [y_coords.copy()]
    boundary_dist_all = [boundary_dist.copy()]
    boundary_zone_all = [boundary_zone.copy()]

    coords_yflip = coords.copy()
    y_max_norm = FIELD_Y / MAX_LEN
    valid_mask = ~padding_mask
    coords_yflip[:, :, 1] = np.where(valid_mask, y_max_norm - coords_yflip[:, :, 1], coords_yflip[:, :, 1])
    coords_yflip[:, :, 3] = np.where(valid_mask, y_max_norm - coords_yflip[:, :, 3], coords_yflip[:, :, 3])

    cont_yflip = cont.copy()
    if "dy" in cont_idx:
        cont_yflip[:, :, cont_idx["dy"]] = -cont_yflip[:, :, cont_idx["dy"]]
    if "action_lateral" in cont_idx:
        cont_yflip[:, :, cont_idx["action_lateral"]] = -cont_yflip[:, :, cont_idx["action_lateral"]]

    angles_yflip = angles.copy()
    for feat in ANGLE_FLIP_FEATURES:
        if feat in angle_idx:
            idx = angle_idx[feat] * 2
            angles_yflip[:, :, idx] = -angles_yflip[:, :, idx]

    cat_yflip = cat.copy()
    if "lane" in cat_idx:
        lane = cat_yflip[:, :, cat_idx["lane"]].copy()
        cat_yflip[:, :, cat_idx["lane"]] = np.where(lane == 0, 2, np.where(lane == 2, 0, lane))

    y_coords_yflip = y_coords.copy()
    y_coords_yflip[:, 1] = FIELD_Y - y_coords_yflip[:, 1]

    zone_yflip = boundary_zone.copy()
    zone_yflip = np.where(zone_yflip == 1, 2, np.where(zone_yflip == 2, 1, zone_yflip))

    coords_all.append(coords_yflip)
    cont_all.append(cont_yflip)
    angles_all.append(angles_yflip)
    cat_all.append(cat_yflip)
    mask_all.append(padding_mask.copy())
    cont_valid_all.append(cont_valid_mask.copy())
    y_coords_all.append(y_coords_yflip)
    boundary_dist_all.append(boundary_dist.copy())
    boundary_zone_all.append(zone_yflip)

    return (
        np.concatenate(coords_all),
        np.concatenate(cont_all),
        np.concatenate(angles_all),
        np.concatenate(cat_all),
        np.concatenate(mask_all),
        np.concatenate(cont_valid_all),
        np.concatenate(y_coords_all),
        np.concatenate(boundary_dist_all),
        np.concatenate(boundary_zone_all),
    )
