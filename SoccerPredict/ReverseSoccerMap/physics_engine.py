"""
Physics-Based Pitch Control Engine
Based on Spearman's "Beyond Expected Goals" (MIT Sloan 2018)

핵심 원리: "누가 먼저 도착하느냐" (Time-to-Intercept)
- 단순 거리가 아닌 시간 기반 점유 경쟁
- 미분 가능한 PyTorch 구현으로 역전파 지원
"""

import torch
import torch.nn as nn
import numpy as np


class PhysicsPitchControl(nn.Module):
    """
    물리 기반 피치 컨트롤 엔진

    경기장의 각 지점에 대해 "공을 여기로 차면 누가 먼저 잡나?"를 계산

    적용 대상: 마지막 패스 이전의 K-1개 패스 이벤트
    - 이전 패스들의 성공/실패로 수비수 위치 추론
    - 추론된 수비 배치로 마지막 패스 예측에 활용
    """

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        max_player_speed: float = 7.5,      # m/s (수비수 최대 속도)
        reaction_time: float = 0.5,          # 초 (반응 시간)
        lambda_coef: float = 4.3,            # 시그모이드 민감도 (Spearman 권장)
    ):
        super().__init__()

        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.max_player_speed = max_player_speed
        self.reaction_time = reaction_time
        self.lambda_coef = lambda_coef

        # 골대 위치 (상대편 골대)
        self.goal_pos = torch.tensor([pitch_length, pitch_width / 2])

    def calculate_player_arrival_time(
        self,
        player_positions: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        선수들이 특정 지점에 도착하는 시간 계산

        T_arrive = T_reaction + Distance / V_max

        Args:
            player_positions: (N, 2) 선수들의 현재 위치
            target: (2,) 목표 지점

        Returns:
            (N,) 각 선수의 도착 시간
        """
        # 거리 계산
        distances = torch.norm(player_positions - target, dim=-1)

        # 시간 = 반응시간 + (거리 / 속도)
        arrival_times = self.reaction_time + (distances / self.max_player_speed)

        return arrival_times

    @staticmethod
    def calculate_ball_speed(
        ball_start: torch.Tensor,
        ball_end: torch.Tensor,
        time_delta: torch.Tensor
    ) -> torch.Tensor:
        """
        실제 데이터에서 공의 속도 계산

        Args:
            ball_start: (2,) 공의 시작 위치
            ball_end: (2,) 공의 종료 위치
            time_delta: 이벤트 간 시간 차이 (초)

        Returns:
            공의 속도 (m/s)
        """
        distance = torch.norm(ball_end - ball_start)
        # 시간이 0이면 기본값 15 m/s 사용
        speed = torch.where(
            time_delta > 0.01,
            distance / time_delta,
            torch.tensor(15.0, device=distance.device)
        )
        return speed

    def calculate_ball_travel_time(
        self,
        ball_start: torch.Tensor,
        ball_end: torch.Tensor,
        ball_speed: torch.Tensor = None,
        time_delta: torch.Tensor = None
    ) -> torch.Tensor:
        """
        공이 시작점에서 목표점까지 이동하는 시간

        Args:
            ball_start: (2,) 공의 시작 위치
            ball_end: (2,) 공의 목표 위치
            ball_speed: 공의 속도 (m/s) - 직접 제공
            time_delta: 실제 시간 차이 - 이게 있으면 그대로 사용

        Returns:
            스칼라 - 공의 이동 시간
        """
        # 실제 time_delta가 있으면 그대로 반환
        if time_delta is not None:
            return time_delta

        # 아니면 속도로 계산
        distance = torch.norm(ball_end - ball_start)
        if ball_speed is None:
            ball_speed = torch.tensor(15.0)  # 기본값
        travel_time = distance / ball_speed
        return travel_time

    def calculate_intercept_probability(
        self,
        ball_time: torch.Tensor,
        defender_times: torch.Tensor
    ) -> torch.Tensor:
        """
        수비수가 패스를 인터셉트할 확률 계산

        P = sigmoid(-λ * (T_ball - min(T_defenders)))

        - T_ball >> T_defender: 수비가 먼저 도착 → P ≈ 1
        - T_ball << T_defender: 공이 먼저 통과 → P ≈ 0

        Args:
            ball_time: 공의 이동 시간
            defender_times: (N,) 수비수들의 도착 시간

        Returns:
            인터셉트 확률 (0~1)
        """
        # 가장 빨리 도착하는 수비수
        min_defender_time = torch.min(defender_times)

        # 시간 차이
        time_diff = ball_time - min_defender_time

        # 수비수가 빨리 도착하면(time_diff > 0) 확률 높음
        prob = torch.sigmoid(self.lambda_coef * (-time_diff))

        return prob

    def forward(
        self,
        ball_start: torch.Tensor,
        ball_end: torch.Tensor,
        defender_positions: torch.Tensor,
        time_delta: torch.Tensor = None
    ) -> torch.Tensor:
        """
        패스에 대한 인터셉트 확률 계산

        Args:
            ball_start: (2,) 공의 시작 위치
            ball_end: (2,) 공의 목표 위치
            defender_positions: (N, 2) 수비수 위치
            time_delta: 실제 패스 시간 (데이터에서 계산됨)

        Returns:
            인터셉트 확률
        """
        # 공의 이동 시간 (실제 데이터가 있으면 사용)
        ball_time = self.calculate_ball_travel_time(
            ball_start, ball_end, time_delta=time_delta
        )

        # 수비수들이 패스 목표 지점에 도착하는 시간
        defender_times = self.calculate_player_arrival_time(defender_positions, ball_end)

        # 인터셉트 확률
        prob = self.calculate_intercept_probability(ball_time, defender_times)

        return prob

    def compute_pitch_control_surface(
        self,
        ball_position: torch.Tensor,
        defender_positions: torch.Tensor,
        grid_resolution: int = 50
    ) -> torch.Tensor:
        """
        전체 경기장에 대한 피치 컨트롤 맵 생성

        Args:
            ball_position: (2,) 현재 공 위치
            defender_positions: (N, 2) 수비수 위치
            grid_resolution: 그리드 해상도

        Returns:
            (H, W) 피치 컨트롤 맵 (0=공격 유리, 1=수비 유리)
        """
        device = ball_position.device

        # 그리드 생성
        x = torch.linspace(0, self.pitch_length, grid_resolution, device=device)
        y = torch.linspace(0, self.pitch_width, grid_resolution, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        # (H, W, 2) 형태의 그리드 포인트
        grid_points = torch.stack([grid_x, grid_y], dim=-1)

        # 각 그리드 포인트에 대해 피치 컨트롤 계산
        H, W = grid_resolution, grid_resolution
        control_map = torch.zeros(H, W, device=device)

        for i in range(H):
            for j in range(W):
                target = grid_points[i, j]
                prob = self.forward(ball_position, target, defender_positions)
                control_map[i, j] = prob

        return control_map


class BatchPhysicsPitchControl(PhysicsPitchControl):
    """
    배치 처리를 지원하는 물리 엔진 (대량 데이터 처리용)
    """

    def forward_batch(
        self,
        ball_starts: torch.Tensor,
        ball_ends: torch.Tensor,
        defender_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        배치 단위로 인터셉트 확률 계산

        Args:
            ball_starts: (B, 2) 공의 시작 위치들
            ball_ends: (B, 2) 공의 목표 위치들
            defender_positions: (B, N, 2) 수비수 위치들

        Returns:
            (B,) 인터셉트 확률들
        """
        batch_size = ball_starts.shape[0]

        # 공의 이동 시간
        ball_distances = torch.norm(ball_ends - ball_starts, dim=-1)
        ball_times = ball_distances / self.ball_speed

        # 수비수들의 도착 시간
        # defender_positions: (B, N, 2), ball_ends: (B, 2) -> (B, 1, 2)
        ball_ends_expanded = ball_ends.unsqueeze(1)
        defender_distances = torch.norm(defender_positions - ball_ends_expanded, dim=-1)
        defender_times = self.reaction_time + (defender_distances / self.max_player_speed)

        # 가장 빨리 도착하는 수비수
        min_defender_times = torch.min(defender_times, dim=-1)[0]

        # 인터셉트 확률
        time_diffs = ball_times - min_defender_times
        probs = torch.sigmoid(self.lambda_coef * (-time_diffs))

        return probs


if __name__ == "__main__":
    # 테스트
    print("Physics Pitch Control Engine Test")
    print("=" * 50)

    engine = PhysicsPitchControl()

    # 시나리오: (50, 34)에서 (80, 40)으로 패스
    ball_start = torch.tensor([50.0, 34.0])
    ball_end = torch.tensor([80.0, 40.0])

    # 수비수 4명 배치
    defenders = torch.tensor([
        [60.0, 30.0],  # 패스 경로 근처
        [70.0, 45.0],  # 목표 지점 근처
        [55.0, 50.0],  # 측면
        [65.0, 20.0],  # 반대편
    ])

    # 인터셉트 확률 계산
    prob = engine(ball_start, ball_end, defenders)

    print(f"Ball: {ball_start.tolist()} -> {ball_end.tolist()}")
    print(f"Defenders: {defenders.tolist()}")
    print(f"Intercept Probability: {prob.item():.4f}")

    # 공 이동 시간
    ball_time = engine.calculate_ball_travel_time(ball_start, ball_end)
    print(f"Ball travel time: {ball_time.item():.2f} sec")

    # 수비수 도착 시간
    defender_times = engine.calculate_player_arrival_time(defenders, ball_end)
    print(f"Defender arrival times: {defender_times.tolist()}")
