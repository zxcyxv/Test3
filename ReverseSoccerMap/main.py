"""
Physics-Based Feature Pipeline
실제 데이터에서 물리 피처 생성

파이프라인:
1. 에피소드별 K-1개 패스 이벤트 추출
2. Ghost Reconstructor로 수비수 위치 복원
3. 복원된 수비수로 물리 피처 생성
4. 기존 피처와 결합
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict

from physics_engine import PhysicsPitchControl
from ghost_reconstructor import GhostDefenderReconstructor
from feature_generator import PhysicsFeatureGenerator


DATA_DIR = Path('/workspace/SoccerPredict/open_track1')


def load_train_data() -> pd.DataFrame:
    """훈련 데이터 로드"""
    return pd.read_csv(DATA_DIR / 'train.csv')


def extract_episode_passes(episode_df: pd.DataFrame) -> List[Dict]:
    """
    에피소드에서 패스 이벤트 추출 (마지막 제외)

    Returns:
        패스 이벤트 리스트 (K-1개)
    """
    # 마지막 이벤트 제외
    df = episode_df.iloc[:-1].copy()

    # Pass 타입만
    passes = df[df['type_name'] == 'Pass']

    events = []
    prev_time = None

    for _, row in passes.iterrows():
        # 시간 차이 계산
        current_time = row['time_seconds']
        if prev_time is not None:
            time_delta = current_time - prev_time
        else:
            time_delta = 1.0  # 첫 패스는 기본값

        event = {
            'start_x': float(row['start_x']),
            'start_y': float(row['start_y']),
            'end_x': float(row['end_x']),
            'end_y': float(row['end_y']),
            'time_delta': max(time_delta, 0.1),  # 최소 0.1초
            'success': row['result_name'] == 'Successful'
        }
        events.append(event)
        prev_time = current_time

    return events


def get_last_pass_start(episode_df: pd.DataFrame) -> tuple:
    """마지막 패스의 시작 좌표 반환"""
    last_row = episode_df.iloc[-1]
    return (float(last_row['start_x']), float(last_row['start_y']))


def process_single_episode(
    episode_df: pd.DataFrame,
    reconstructor: GhostDefenderReconstructor,
    feature_gen: PhysicsFeatureGenerator,
    n_iterations: int = 30
) -> Dict[str, float]:
    """
    단일 에피소드 처리

    Args:
        episode_df: 에피소드 데이터프레임
        reconstructor: 수비수 복원기
        feature_gen: 피처 생성기
        n_iterations: 복원 반복 횟수

    Returns:
        물리 피처 딕셔너리
    """
    # 1. 패스 이벤트 추출
    pass_events = extract_episode_passes(episode_df)

    # 2. 마지막 패스 시작 위치
    last_start = get_last_pass_start(episode_df)
    last_start_tensor = torch.tensor(last_start)

    # 3. 수비수 위치 복원
    if len(pass_events) >= 1:
        # 최근 3개 패스만 사용 (효율성)
        recent_events = pass_events[-3:] if len(pass_events) > 3 else pass_events
        defender_coords = reconstructor.reconstruct_from_sequence(
            recent_events,
            n_iterations=n_iterations,
            verbose=False
        )
    else:
        # 패스가 없으면 기본 위치
        defender_coords = torch.tensor([
            [60.0, 25.0], [70.0, 35.0], [65.0, 45.0], [75.0, 30.0]
        ])

    # 4. 피처 생성
    base_features = feature_gen.compute_features(last_start_tensor, defender_coords)
    boundary_features = feature_gen.compute_features_for_boundary_prediction(
        last_start_tensor, defender_coords
    )

    # 5. 수비수 좌표도 피처로 추가
    for i, coord in enumerate(defender_coords):
        base_features[f'defender_{i}_x'] = coord[0].item()
        base_features[f'defender_{i}_y'] = coord[1].item()

    return {**base_features, **boundary_features}


def generate_physics_features(
    df: pd.DataFrame,
    n_defenders: int = 4,
    n_iterations: int = 30,
    sample_size: int = None
) -> pd.DataFrame:
    """
    전체 데이터에 대해 물리 피처 생성

    Args:
        df: 원본 이벤트 데이터
        n_defenders: 복원할 수비수 수
        n_iterations: 복원 반복 횟수
        sample_size: 샘플링 (테스트용)

    Returns:
        에피소드별 물리 피처 DataFrame
    """
    # 초기화
    reconstructor = GhostDefenderReconstructor(num_defenders=n_defenders)
    feature_gen = PhysicsFeatureGenerator()

    # 에피소드별 그룹화
    episodes = df.groupby('game_episode')

    if sample_size:
        episode_ids = list(episodes.groups.keys())[:sample_size]
    else:
        episode_ids = list(episodes.groups.keys())

    print(f"Processing {len(episode_ids)} episodes...")

    all_features = []

    for ep_id in tqdm(episode_ids):
        episode_df = episodes.get_group(ep_id).sort_values('action_id')

        try:
            features = process_single_episode(
                episode_df,
                reconstructor,
                feature_gen,
                n_iterations
            )
            features['game_episode'] = ep_id
            all_features.append(features)
        except Exception as e:
            print(f"Error processing {ep_id}: {e}")
            continue

    # DataFrame으로 변환
    features_df = pd.DataFrame(all_features)

    return features_df


def merge_with_existing_features(
    physics_features: pd.DataFrame,
    existing_features_path: Path
) -> pd.DataFrame:
    """
    물리 피처를 기존 피처와 병합

    Args:
        physics_features: 물리 피처 DataFrame
        existing_features_path: 기존 피처 CSV 경로

    Returns:
        병합된 DataFrame
    """
    existing = pd.read_csv(existing_features_path)

    # game_episode 기준으로 병합
    merged = existing.merge(
        physics_features,
        on='game_episode',
        how='left'
    )

    return merged


def main():
    """메인 실행"""
    print("=" * 60)
    print("Physics-Based Feature Generation Pipeline")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1] Loading data...")
    df = load_train_data()
    print(f"Loaded {len(df)} events, {df['game_episode'].nunique()} episodes")

    # 2. 물리 피처 생성 (테스트용으로 100개만)
    print("\n[2] Generating physics features (sample)...")
    physics_features = generate_physics_features(
        df,
        n_defenders=4,
        n_iterations=30,
        sample_size=100  # 테스트용
    )
    print(f"Generated {len(physics_features)} feature rows")
    print(f"Feature columns: {list(physics_features.columns)[:10]}...")

    # 3. 저장
    output_path = DATA_DIR / 'physics_features_sample.csv'
    physics_features.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # 4. 피처 통계
    print("\n[3] Feature Statistics:")
    numeric_cols = physics_features.select_dtypes(include=[np.number]).columns
    print(physics_features[numeric_cols].describe().T[['mean', 'std', 'min', 'max']])

    return physics_features


if __name__ == '__main__':
    features = main()
