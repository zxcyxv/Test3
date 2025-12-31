"""
Feature Engineering V2: 압박 및 기하학적 지표

Data Leakage를 방지하면서 모델 성능을 극대화하는 피처 엔지니어링

피처 카테고리:
1. 정적/기하학적 피처 (Static & Geometric)
   - pressure_x_weight: 필드 틸트 & 압박 가중치
   - is_zone14: Zone 14 및 위험 구역 플래그
   - dist_to_goal: 골대까지 거리
   - angle_visible: 골대 시야각

2. 동적/시퀀스 피처 (Dynamic Sequence)
   - action_angle: 액션 방향 각도
   - action_progress: 전진 거리
   - action_dist: 이동 거리
   - action_lateral: 횡방향 이동
"""

import pandas as pd
import numpy as np
from pathlib import Path

# =============================================================================
# Constants
# =============================================================================

FIELD_X = 105.0
FIELD_Y = 68.0
GOAL_X = 105.0
GOAL_Y = 34.0  # 골대 중앙 (68 / 2)
EPSILON = 1e-6

K = 8  # 시퀀스 길이

DATA_DIR = Path('/workspace/SoccerPredict/open_track1')


# =============================================================================
# 2. 정적/기하학적 피처 (Static & Geometric Features)
# =============================================================================

def add_geometric_features(df: pd.DataFrame, k: int = K) -> pd.DataFrame:
    """
    정적/기하학적 피처 추가

    적용 대상: 모든 K개 이벤트
    입력: start_x, start_y

    Args:
        df: 입력 DataFrame
        k: 시퀀스 길이

    Returns:
        피처가 추가된 DataFrame
    """
    df = df.copy()

    for i in range(k):
        sx_col = f'start_x_{i}'
        sy_col = f'start_y_{i}'

        if sx_col not in df.columns:
            print(f"  Warning: {sx_col} not found, skipping index {i}")
            continue

        sx = df[sx_col]
        sy = df[sy_col]

        # ---------------------------------------------------------------------
        # 2.1 Pressure X Weight (Field Tilt Proxy)
        # 공이 상대 진영 깊숙이 들어갈수록 비선형적으로 증가
        # 하프라인(52.5): 0.25, 페널티박스(85): 0.65, 골라인(105): 1.0
        # ---------------------------------------------------------------------
        df[f'pressure_x_weight_{i}'] = (sx / FIELD_X) ** 2

        # ---------------------------------------------------------------------
        # 2.2 Zone 14 Flag
        # 가장 위협적인 중앙 지역 식별
        # X: 70 ~ 88.5 (파이널 서드 진입 ~ 페널티박스 앞)
        # Y: 24 ~ 44 (중앙)
        # ---------------------------------------------------------------------
        is_zone14 = (
            (sx >= 70) & (sx <= 88.5) &
            (sy >= 24) & (sy <= 44)
        )
        df[f'is_zone14_{i}'] = is_zone14.astype(int)

        # ---------------------------------------------------------------------
        # 2.3 Distance to Goal
        # 골대 중심(105, 34)까지의 유클리드 거리
        # ---------------------------------------------------------------------
        df[f'dist_to_goal_{i}'] = np.sqrt(
            (GOAL_X - sx) ** 2 + (GOAL_Y - sy) ** 2
        )

        # ---------------------------------------------------------------------
        # 2.4 Angle Visible (Goal Centrality)
        # 골대 중심을 바라보는 각도 (0에 가까울수록 정면)
        # ---------------------------------------------------------------------
        df[f'angle_visible_{i}'] = np.arctan(
            np.abs(GOAL_Y - sy) / (GOAL_X - sx + EPSILON)
        )

        # ---------------------------------------------------------------------
        # 2.5 Polar Angle (θ) - 골대 중심 기준 극좌표 각도
        # arctan2(dy, dx): -π ~ π 범위, 부호 정보 보존
        # 골대를 원점으로 했을 때의 방향 정보
        # ---------------------------------------------------------------------
        df[f'polar_angle_{i}'] = np.arctan2(GOAL_Y - sy, GOAL_X - sx)

        # ---------------------------------------------------------------------
        # 2.6 Log Distance to Goal
        # 골라인 근처의 미세한 변화 감지를 위한 로그 스케일 거리
        # 경계선 부근의 해상도를 높임
        # ---------------------------------------------------------------------
        df[f'log_dist_to_goal_{i}'] = np.log(df[f'dist_to_goal_{i}'] + 1)

    return df


# =============================================================================
# 3. 동적/시퀀스 피처 (Dynamic Sequence Features)
# =============================================================================

def add_sequence_features(df: pd.DataFrame, k: int = K) -> pd.DataFrame:
    """
    동적/시퀀스 피처 추가

    적용 대상: 직전 액션들 (0 ~ k-2)
    입력: start_x, start_y, end_x, end_y

    Args:
        df: 입력 DataFrame
        k: 시퀀스 길이

    Returns:
        피처가 추가된 DataFrame
    """
    df = df.copy()

    # -------------------------------------------------------------------------
    # 3.1 & 3.2 각 이벤트별 방향 및 전진성 피처
    # -------------------------------------------------------------------------
    for i in range(k - 1):  # 0 ~ k-2 (마지막 이벤트 제외)
        sx_col = f'start_x_{i}'
        sy_col = f'start_y_{i}'
        ex_col = f'end_x_{i}'
        ey_col = f'end_y_{i}'

        # end_x, end_y는 마스킹된 피처 (k-1은 없음)
        if ex_col not in df.columns:
            print(f"  Warning: {ex_col} not found, skipping index {i}")
            continue

        sx = df[sx_col]
        sy = df[sy_col]
        ex = df[ex_col]
        ey = df[ey_col]

        # Action Angle (방향)
        # atan2(dy, dx): -π ~ π 범위의 각도
        df[f'action_angle_{i}'] = np.arctan2(ey - sy, ex - sx)

        # Action Progress (전진 거리)
        # 양수: 전진, 음수: 백패스
        df[f'action_progress_{i}'] = ex - sx

        # Action Distance (총 이동 거리)
        df[f'action_dist_{i}'] = np.sqrt((ex - sx) ** 2 + (ey - sy) ** 2)

        # Action Lateral (횡방향 이동)
        # 양수: 위쪽, 음수: 아래쪽
        df[f'action_lateral_{i}'] = ey - sy

    # -------------------------------------------------------------------------
    # 직전 액션 피처 (t-1) - 명세서 요구사항
    # -------------------------------------------------------------------------
    last_hist_idx = k - 2  # 인덱스 6 (마지막 직전)

    df['prev_action_angle'] = df.get(f'action_angle_{last_hist_idx}', 0)
    df['prev_action_progress'] = df.get(f'action_progress_{last_hist_idx}', 0)
    df['prev_action_dist'] = df.get(f'action_dist_{last_hist_idx}', 0)
    df['prev_action_lateral'] = df.get(f'action_lateral_{last_hist_idx}', 0)

    # -------------------------------------------------------------------------
    # 최근 N개 액션 집계 피처
    # -------------------------------------------------------------------------
    recent_indices = [k-2, k-3, k-4]  # 6, 5, 4 (최근 3개)

    # Progress 집계
    progress_cols = [
        f'action_progress_{i}'
        for i in recent_indices
        if f'action_progress_{i}' in df.columns
    ]
    if progress_cols:
        df['recent_progress_sum'] = df[progress_cols].sum(axis=1)
        df['recent_progress_mean'] = df[progress_cols].mean(axis=1)
        df['recent_progress_std'] = df[progress_cols].std(axis=1)

    # Distance 집계
    dist_cols = [
        f'action_dist_{i}'
        for i in recent_indices
        if f'action_dist_{i}' in df.columns
    ]
    if dist_cols:
        df['recent_dist_sum'] = df[dist_cols].sum(axis=1)
        df['recent_dist_mean'] = df[dist_cols].mean(axis=1)

    # Angle 변화량 (방향 전환 정도)
    angle_cols = [
        f'action_angle_{i}'
        for i in recent_indices
        if f'action_angle_{i}' in df.columns
    ]
    if len(angle_cols) >= 2:
        df['recent_angle_std'] = df[angle_cols].std(axis=1)

    # Lateral 집계 (횡방향 움직임)
    lateral_cols = [
        f'action_lateral_{i}'
        for i in recent_indices
        if f'action_lateral_{i}' in df.columns
    ]
    if lateral_cols:
        df['recent_lateral_sum'] = df[lateral_cols].sum(axis=1)
        df['recent_lateral_abs_sum'] = df[lateral_cols].abs().sum(axis=1)

    return df


# =============================================================================
# 마지막 액션 강조 피처
# =============================================================================

def add_last_action_features(df: pd.DataFrame, k: int = K) -> pd.DataFrame:
    """
    마지막 액션(예측 대상)의 핵심 피처를 별도 컬럼으로 강조

    Args:
        df: 입력 DataFrame
        k: 시퀀스 길이

    Returns:
        피처가 추가된 DataFrame
    """
    df = df.copy()

    last_idx = k - 1  # 인덱스 7
    sx = df[f'start_x_{last_idx}']
    sy = df[f'start_y_{last_idx}']

    # 핵심 좌표
    df['last_start_x'] = sx
    df['last_start_y'] = sy

    # 압박 지표
    df['last_pressure'] = (sx / FIELD_X) ** 2

    # 골대 관련
    df['last_dist_to_goal'] = np.sqrt(
        (GOAL_X - sx) ** 2 + (GOAL_Y - sy) ** 2
    )
    df['last_angle_visible'] = np.arctan(
        np.abs(GOAL_Y - sy) / (GOAL_X - sx + EPSILON)
    )

    # Zone 14
    df['last_is_zone14'] = (
        (sx >= 70) & (sx <= 88.5) &
        (sy >= 24) & (sy <= 44)
    ).astype(int)

    # Polar Coordinates
    df['last_polar_angle'] = np.arctan2(GOAL_Y - sy, GOAL_X - sx)
    df['last_log_dist_to_goal'] = np.log(df['last_dist_to_goal'] + 1)

    # 필드 영역 분류 (3x3 그리드)
    df['last_field_third'] = pd.cut(
        sx,
        bins=[-np.inf, 35, 70, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    df['last_y_zone'] = pd.cut(
        sy,
        bins=[-np.inf, 22.67, 45.33, np.inf],
        labels=[0, 1, 2]
    ).astype(int)

    return df


# =============================================================================
# 메인 실행
# =============================================================================

def process_dataset(df: pd.DataFrame, name: str = "Dataset") -> pd.DataFrame:
    """
    전체 피처 엔지니어링 파이프라인 실행

    Args:
        df: 입력 DataFrame
        name: 데이터셋 이름 (로깅용)

    Returns:
        피처가 추가된 DataFrame
    """
    print(f"\n  Processing {name}...")
    original_cols = len(df.columns)

    print(f"    - Adding geometric features...")
    df = add_geometric_features(df)

    print(f"    - Adding sequence features...")
    df = add_sequence_features(df)

    print(f"    - Adding last action features...")
    df = add_last_action_features(df)

    new_cols = len(df.columns) - original_cols
    print(f"    - Added {new_cols} new features ({original_cols} -> {len(df.columns)})")

    return df


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Feature Engineering V2: 압박 및 기하학적 지표")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1] Loading data...")
    train_df = pd.read_csv(DATA_DIR / 'train_features_k8.csv')
    test_df = pd.read_csv(DATA_DIR / 'test_features_k8.csv')
    print(f"  Train: {len(train_df)} rows, {len(train_df.columns)} cols")
    print(f"  Test: {len(test_df)} rows, {len(test_df.columns)} cols")

    # 2. 피처 생성
    print("\n[2] Generating features...")
    train_df = process_dataset(train_df, "Train")
    test_df = process_dataset(test_df, "Test")

    # 3. 새 피처 목록 출력
    print("\n[3] New features summary:")
    original_cols = set(pd.read_csv(DATA_DIR / 'train_features_k8.csv').columns)
    new_cols = [c for c in train_df.columns if c not in original_cols]

    geo_cols = [c for c in new_cols if any(x in c for x in ['pressure', 'zone14', 'dist_to_goal', 'angle_visible'])]
    seq_cols = [c for c in new_cols if any(x in c for x in ['action_', 'prev_', 'recent_'])]
    last_cols = [c for c in new_cols if c.startswith('last_')]

    print(f"\n  [Geometric Features] {len(geo_cols)} features")
    print(f"  [Sequence Features] {len(seq_cols)} features")
    print(f"  [Last Action Features] {len(last_cols)} features")
    print(f"  [Total New Features] {len(new_cols)} features")

    # 4. 저장
    print("\n[4] Saving...")
    train_output = DATA_DIR / 'train_features_v2.csv'
    test_output = DATA_DIR / 'test_features_v2.csv'

    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)

    print(f"  Train: {train_output}")
    print(f"  Test: {test_output}")

    # 5. 통계 출력
    print("\n[5] Feature statistics (Train):")
    stat_cols = [
        'last_pressure',
        'last_dist_to_goal',
        'last_angle_visible',
        'last_is_zone14',
        'prev_action_progress',
        'recent_progress_sum'
    ]
    stat_cols = [c for c in stat_cols if c in train_df.columns]
    print(train_df[stat_cols].describe().round(3))

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    return train_df, test_df


if __name__ == '__main__':
    train_df, test_df = main()
