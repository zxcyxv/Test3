from pathlib import Path

DATA_DIR = Path("/workspace/SoccerPredict/open_track1")
FIELD_X, FIELD_Y = 105, 68
K = 8
MAX_LEN = 105.0

COORD_FEATURES = ["start_x", "start_y", "end_x", "end_y"]
ANGLE_FEATURES = ["angle_to_goal", "action_angle", "angle_visible"]
ANGLE_FLIP_FEATURES = ["angle_to_goal", "action_angle"]
CAT_FEATURES = ["type_id", "res_id", "is_home", "x_zone", "lane", "is_zone14"]
CONT_FEATURES = [
    "dt",
    "ep_idx_norm",
    "dist_to_goal",
    "pressure_x_weight",
    "dx",
    "dy",
    "dist",
    "speed",
    "action_progress",
    "action_dist",
    "action_lateral",
]
