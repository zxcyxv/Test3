"""
Feature Generator
복원된 수비수 위치로부터 마지막 패스 예측용 피처 생성

생성 피처:
1. min_defender_dist: 마지막 패스 시작점에서 가장 가까운 수비수 거리
2. defender_density_start: 시작점 주변 수비수 밀집도
3. defender_density_end_zone: 예상 종료 구역별 수비 밀집도
4. pass_lane_blocked: 각 방향으로의 패스 경로 차단 정도
5. intercept_prob_*: 여러 방향에 대한 인터셉트 확률
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from physics_engine import PhysicsPitchControl


class PhysicsFeatureGenerator:
    """
    물리 엔진 기반 피처 생성기

    복원된 수비수 위치를 사용하여 마지막 패스 예측에 도움되는 피처 생성
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ):
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.physics = PhysicsPitchControl(
            pitch_length=pitch_length,
            pitch_width=pitch_width
        )

        # 피치를 구역으로 나눔 (3x3 = 9구역)
        self.x_zones = 3
        self.y_zones = 3

    def compute_features(
        self,
        ball_start: torch.Tensor,
        defender_positions: torch.Tensor,
    ) -> Dict[str, float]:
        """
        마지막 패스 시작 위치와 수비수 배치로부터 피처 계산

        Args:
            ball_start: (2,) 마지막 패스 시작 위치
            defender_positions: (N, 2) 복원된 수비수 위치

        Returns:
            피처 딕셔너리
        """
        features = {}

        # 1. 기본 거리 피처
        distances = torch.norm(defender_positions - ball_start, dim=-1)
        features['min_defender_dist'] = distances.min().item()
        features['mean_defender_dist'] = distances.mean().item()
        features['std_defender_dist'] = distances.std().item()

        # 2. 시작점 주변 밀집도 (5m, 10m 반경)
        features['defenders_within_5m'] = (distances < 5).sum().item()
        features['defenders_within_10m'] = (distances < 10).sum().item()

        # 3. 골대 방향 수비 밀집도
        goal_pos = torch.tensor([self.pitch_length, self.pitch_width / 2])
        goal_direction = goal_pos - ball_start
        goal_direction = goal_direction / torch.norm(goal_direction)

        # 골대 방향 원뿔 내 수비수 수
        defender_directions = defender_positions - ball_start
        defender_distances = torch.norm(defender_directions, dim=-1)

        # 각도 계산 (골대 방향과의 각도)
        cos_angles = torch.sum(
            defender_directions * goal_direction, dim=-1
        ) / (defender_distances + 1e-6)

        # 30도 이내 수비수
        features['defenders_in_goal_cone_30'] = (cos_angles > 0.866).sum().item()  # cos(30°)
        # 45도 이내 수비수
        features['defenders_in_goal_cone_45'] = (cos_angles > 0.707).sum().item()  # cos(45°)

        # 4. 구역별 수비 밀집도
        zone_density = self._compute_zone_density(defender_positions)
        for i, density in enumerate(zone_density):
            features[f'zone_{i}_density'] = density

        # 5. 패스 방향별 인터셉트 확률 (8방향)
        intercept_probs = self._compute_directional_intercept_probs(
            ball_start, defender_positions
        )
        for direction, prob in intercept_probs.items():
            features[f'intercept_prob_{direction}'] = prob

        # 6. 가장 열린 방향
        min_prob_direction = min(intercept_probs, key=intercept_probs.get)
        features['best_pass_direction'] = list(intercept_probs.keys()).index(min_prob_direction)
        features['best_pass_intercept_prob'] = intercept_probs[min_prob_direction]

        # 7. 수비 라인 높이 (수비수들의 평균 x좌표)
        features['defense_line_x'] = defender_positions[:, 0].mean().item()

        # 8. 수비 폭 (수비수들의 y좌표 분산)
        features['defense_width'] = defender_positions[:, 1].std().item()

        return features

    def _compute_zone_density(
        self,
        defender_positions: torch.Tensor
    ) -> List[float]:
        """
        경기장을 9구역으로 나누고 각 구역의 수비수 밀집도 계산
        """
        zone_width = self.pitch_length / self.x_zones
        zone_height = self.pitch_width / self.y_zones

        densities = []
        for i in range(self.x_zones):
            for j in range(self.y_zones):
                x_min, x_max = i * zone_width, (i + 1) * zone_width
                y_min, y_max = j * zone_height, (j + 1) * zone_height

                in_zone = (
                    (defender_positions[:, 0] >= x_min) &
                    (defender_positions[:, 0] < x_max) &
                    (defender_positions[:, 1] >= y_min) &
                    (defender_positions[:, 1] < y_max)
                )
                densities.append(in_zone.sum().item())

        return densities

    def _compute_directional_intercept_probs(
        self,
        ball_start: torch.Tensor,
        defender_positions: torch.Tensor,
        pass_distance: float = 20.0
    ) -> Dict[str, float]:
        """
        8방향으로 패스했을 때의 인터셉트 확률 계산
        """
        directions = {
            'N': (0, 1),
            'NE': (1, 1),
            'E': (1, 0),
            'SE': (1, -1),
            'S': (0, -1),
            'SW': (-1, -1),
            'W': (-1, 0),
            'NW': (-1, 1),
        }

        probs = {}
        for name, (dx, dy) in directions.items():
            # 정규화된 방향 벡터
            length = np.sqrt(dx**2 + dy**2)
            dx_norm, dy_norm = dx / length, dy / length

            # 목표 지점
            ball_end = ball_start + torch.tensor([dx_norm * pass_distance, dy_norm * pass_distance])

            # 경기장 내로 클리핑
            ball_end[0] = torch.clamp(ball_end[0], 0, self.pitch_length)
            ball_end[1] = torch.clamp(ball_end[1], 0, self.pitch_width)

            # 인터셉트 확률
            with torch.no_grad():
                prob = self.physics(ball_start, ball_end, defender_positions)

            probs[name] = prob.item()

        return probs

    def compute_features_for_boundary_prediction(
        self,
        ball_start: torch.Tensor,
        defender_positions: torch.Tensor,
    ) -> Dict[str, float]:
        """
        경계(Top/Bottom/Goal-line) 예측을 위한 특화 피처

        Args:
            ball_start: 마지막 패스 시작 위치
            defender_positions: 복원된 수비수 위치

        Returns:
            경계 예측용 피처
        """
        features = {}

        # 1. 상단/하단 경계까지의 열린 정도
        # 상단 (y=68) 방향 인터셉트 확률
        top_target = torch.tensor([ball_start[0].item() + 10, 68.0])
        bottom_target = torch.tensor([ball_start[0].item() + 10, 0.0])
        goal_target = torch.tensor([105.0, ball_start[1].item()])

        with torch.no_grad():
            features['intercept_prob_top'] = self.physics(
                ball_start, top_target, defender_positions
            ).item()
            features['intercept_prob_bottom'] = self.physics(
                ball_start, bottom_target, defender_positions
            ).item()
            features['intercept_prob_goal'] = self.physics(
                ball_start, goal_target, defender_positions
            ).item()

        # 2. 각 경계까지의 수비수 수
        # Top boundary (y > 60)
        features['defenders_near_top'] = (defender_positions[:, 1] > 60).sum().item()
        # Bottom boundary (y < 8)
        features['defenders_near_bottom'] = (defender_positions[:, 1] < 8).sum().item()
        # Goal line (x > 100)
        features['defenders_near_goal'] = (defender_positions[:, 0] > 100).sum().item()

        # 3. 경계별 가장 가까운 수비수 거리
        top_edge = torch.tensor([[ball_start[0], 68.0]])
        bottom_edge = torch.tensor([[ball_start[0], 0.0]])
        goal_edge = torch.tensor([[105.0, ball_start[1]]])

        features['min_dist_defender_to_top'] = torch.norm(
            defender_positions - top_edge, dim=-1
        ).min().item()
        features['min_dist_defender_to_bottom'] = torch.norm(
            defender_positions - bottom_edge, dim=-1
        ).min().item()
        features['min_dist_defender_to_goal'] = torch.norm(
            defender_positions - goal_edge, dim=-1
        ).min().item()

        return features


def generate_physics_features_batch(
    episodes_data: List[Dict],
    n_defenders: int = 4,
    n_iterations: int = 30
) -> List[Dict[str, float]]:
    """
    여러 에피소드에 대해 물리 피처 생성

    Args:
        episodes_data: 에피소드별 데이터 리스트
            각 에피소드: {
                'pass_events': K-1개의 패스 이벤트,
                'last_pass_start': 마지막 패스 시작 좌표
            }
        n_defenders: 복원할 수비수 수
        n_iterations: 복원 반복 횟수

    Returns:
        각 에피소드의 물리 피처 리스트
    """
    from ghost_reconstructor import GhostDefenderReconstructor

    reconstructor = GhostDefenderReconstructor(num_defenders=n_defenders)
    feature_gen = PhysicsFeatureGenerator()

    all_features = []

    for episode in episodes_data:
        pass_events = episode['pass_events']
        last_start = torch.tensor(episode['last_pass_start'])

        # 수비수 위치 복원
        if len(pass_events) > 0:
            defender_coords = reconstructor.reconstruct_from_sequence(
                pass_events,
                n_iterations=n_iterations,
                verbose=False
            )
        else:
            # 패스가 없으면 기본 위치 사용
            defender_coords = torch.tensor([
                [60.0, 25.0], [70.0, 35.0], [65.0, 45.0], [75.0, 30.0]
            ])

        # 피처 생성
        base_features = feature_gen.compute_features(last_start, defender_coords)
        boundary_features = feature_gen.compute_features_for_boundary_prediction(
            last_start, defender_coords
        )

        # 합치기
        features = {**base_features, **boundary_features}
        all_features.append(features)

    return all_features


if __name__ == "__main__":
    print("Physics Feature Generator Test")
    print("=" * 50)

    # 테스트용 수비수 배치
    defenders = torch.tensor([
        [70.0, 30.0],
        [75.0, 40.0],
        [80.0, 25.0],
        [85.0, 35.0],
    ])

    # 마지막 패스 시작점
    ball_start = torch.tensor([60.0, 34.0])

    generator = PhysicsFeatureGenerator()

    # 기본 피처
    features = generator.compute_features(ball_start, defenders)
    print("\n기본 피처:")
    for k, v in features.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 경계 예측용 피처
    boundary_features = generator.compute_features_for_boundary_prediction(
        ball_start, defenders
    )
    print("\n경계 예측용 피처:")
    for k, v in boundary_features.items():
        print(f"  {k}: {v:.4f}")
