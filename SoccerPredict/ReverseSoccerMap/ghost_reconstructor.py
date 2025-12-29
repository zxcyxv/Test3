"""
Ghost Defender Reconstructor
패스 성공/실패 결과로부터 수비수 위치를 역추론

핵심 아이디어:
- 마지막 패스 이전의 K-1개 패스 이벤트 활용
- 각 패스의 성공/실패 정보로 수비수 위치 추론
- 추론된 수비 배치를 마지막 패스 예측에 활용
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional
import numpy as np

from physics_engine import PhysicsPitchControl


class GhostDefenderReconstructor(nn.Module):
    """
    패스 결과로부터 수비수 위치를 역추론하는 모듈

    사용법:
    1. 에피소드의 K-1개 패스 이벤트 입력
    2. 각 패스의 성공/실패에 맞게 수비수 좌표 최적화
    3. 최종 수비수 배치 반환
    """

    def __init__(
        self,
        num_defenders: int = 4,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        max_player_speed: float = 7.5,
        reaction_time: float = 0.5,
        lambda_coef: float = 4.3,
    ):
        super().__init__()

        self.num_defenders = num_defenders
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # 물리 엔진
        self.physics = PhysicsPitchControl(
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            max_player_speed=max_player_speed,
            reaction_time=reaction_time,
            lambda_coef=lambda_coef,
        )

        # 수비수 좌표 (학습 가능한 파라미터)
        # 초기화: 경기장 중앙~수비 지역에 랜덤 배치
        init_coords = torch.zeros(num_defenders, 2)
        init_coords[:, 0] = torch.rand(num_defenders) * 40 + 50  # x: 50~90
        init_coords[:, 1] = torch.rand(num_defenders) * 40 + 14  # y: 14~54
        self.coords = nn.Parameter(init_coords)

        # 골대 위치
        self.goal_pos = torch.tensor([pitch_length, pitch_width / 2])

    def forward(
        self,
        ball_start: torch.Tensor,
        ball_end: torch.Tensor,
        time_delta: torch.Tensor = None
    ) -> torch.Tensor:
        """
        현재 수비수 배치에서 인터셉트 확률 계산

        Args:
            ball_start: (2,) 패스 시작 위치
            ball_end: (2,) 패스 종료 위치
            time_delta: 실제 패스 소요 시간

        Returns:
            인터셉트 확률 (0~1)
        """
        return self.physics(ball_start, ball_end, self.coords, time_delta)

    def compute_loss(
        self,
        ball_start: torch.Tensor,
        ball_end: torch.Tensor,
        pass_success: bool,
        time_delta: torch.Tensor = None,
        tactic_weight: float = 0.05
    ) -> torch.Tensor:
        """
        Loss 계산

        구성:
        1. Reality Loss: 패스 결과와 예측의 일치
        2. Tactic Loss: 수비수가 합리적인 위치에 있도록 제약

        Args:
            ball_start: 패스 시작 위치
            ball_end: 패스 종료 위치
            pass_success: True면 패스 성공, False면 실패
            time_delta: 실제 패스 시간
            tactic_weight: 전술 제약 가중치
        """
        # 1. Reality Loss
        prob_intercept = self.forward(ball_start, ball_end, time_delta)

        if pass_success:
            # 패스 성공 → 인터셉트 확률이 낮아야 함
            loss_reality = prob_intercept ** 2
        else:
            # 패스 실패 → 인터셉트 확률이 높아야 함
            loss_reality = (1 - prob_intercept) ** 2

        # 2. Tactic Loss: 수비수들이 골대-공 사이에 있도록
        # 수비수 무게중심
        def_center = torch.mean(self.coords, dim=0)

        # 이상적인 수비 위치: 공과 골대 사이 어딘가
        ideal_pos = (ball_start + self.goal_pos) / 2
        loss_tactic = torch.norm(def_center - ideal_pos) * tactic_weight

        # 3. Boundary Loss: 경기장 밖으로 나가지 않도록 (soft constraint)
        out_of_bounds_x = torch.relu(-self.coords[:, 0]) + torch.relu(self.coords[:, 0] - self.pitch_length)
        out_of_bounds_y = torch.relu(-self.coords[:, 1]) + torch.relu(self.coords[:, 1] - self.pitch_width)
        loss_boundary = (out_of_bounds_x.sum() + out_of_bounds_y.sum()) * 0.1

        return loss_reality + loss_tactic + loss_boundary

    def reconstruct_from_sequence(
        self,
        pass_events: List[dict],
        n_iterations: int = 50,
        lr: float = 1.0,
        verbose: bool = False
    ) -> torch.Tensor:
        """
        패스 시퀀스로부터 수비수 위치 복원

        Args:
            pass_events: 패스 이벤트 리스트
                각 이벤트: {
                    'start_x', 'start_y': 시작 좌표
                    'end_x', 'end_y': 종료 좌표
                    'time_delta': 소요 시간
                    'success': True/False
                }
            n_iterations: 최적화 반복 횟수
            lr: 학습률
            verbose: 진행 상황 출력

        Returns:
            (N, 2) 복원된 수비수 좌표
        """
        # 좌표 초기화 (매번 새로 시작)
        self._reset_coords()

        # Optimizer 설정
        optimizer = optim.Adam([self.coords], lr=lr)

        for iteration in range(n_iterations):
            optimizer.zero_grad()

            total_loss = 0
            for event in pass_events:
                ball_start = torch.tensor([event['start_x'], event['start_y']])
                ball_end = torch.tensor([event['end_x'], event['end_y']])
                time_delta = torch.tensor(event.get('time_delta', 1.0))
                success = event['success']

                loss = self.compute_loss(ball_start, ball_end, success, time_delta)
                total_loss = total_loss + loss

            total_loss.backward()
            optimizer.step()

            # 좌표 클리핑 (경기장 내로 제한)
            with torch.no_grad():
                self.coords[:, 0].clamp_(0, self.pitch_length)
                self.coords[:, 1].clamp_(0, self.pitch_width)

            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iter {iteration+1}: Loss = {total_loss.item():.4f}")

        return self.coords.detach().clone()

    def _reset_coords(self):
        """수비수 좌표 초기화"""
        with torch.no_grad():
            self.coords[:, 0] = torch.rand(self.num_defenders) * 40 + 50
            self.coords[:, 1] = torch.rand(self.num_defenders) * 40 + 14


class SequentialGhostReconstructor:
    """
    시퀀스 단위로 수비수 위치를 추적하는 래퍼

    각 패스 이벤트마다 수비수 위치를 업데이트하면서
    시간에 따른 수비 배치 변화를 추적
    """

    def __init__(
        self,
        num_defenders: int = 4,
        max_player_speed: float = 7.5,
    ):
        self.num_defenders = num_defenders
        self.max_speed = max_player_speed
        self.reconstructor = GhostDefenderReconstructor(num_defenders=num_defenders)

    def process_episode(
        self,
        pass_events: List[dict],
        n_iterations: int = 30
    ) -> List[torch.Tensor]:
        """
        에피소드의 모든 패스를 처리하고 각 시점의 수비 배치 반환

        Args:
            pass_events: 패스 이벤트 리스트 (마지막 패스 제외, K-1개)
            n_iterations: 각 패스당 최적화 반복

        Returns:
            각 패스 시점의 수비수 좌표 리스트
        """
        defender_positions = []

        # 누적 패스로 처리 (이전 패스 정보도 반영)
        for i in range(len(pass_events)):
            # 현재까지의 패스들로 복원
            events_so_far = pass_events[:i+1]

            # 마지막 몇 개만 사용 (메모리/시간 효율)
            recent_events = events_so_far[-3:] if len(events_so_far) > 3 else events_so_far

            coords = self.reconstructor.reconstruct_from_sequence(
                recent_events,
                n_iterations=n_iterations,
                verbose=False
            )
            defender_positions.append(coords)

        return defender_positions


def extract_pass_events_from_episode(episode_df) -> List[dict]:
    """
    에피소드 DataFrame에서 패스 이벤트 추출 (마지막 제외)

    Args:
        episode_df: 에피소드의 이벤트 DataFrame

    Returns:
        패스 이벤트 리스트
    """
    pass_events = []

    # 마지막 이벤트 제외
    df = episode_df.iloc[:-1]

    # Pass 타입만 필터링
    passes = df[df['type_name'] == 'Pass']

    for _, row in passes.iterrows():
        event = {
            'start_x': row['start_x'],
            'start_y': row['start_y'],
            'end_x': row['end_x'],
            'end_y': row['end_y'],
            'time_delta': row.get('dt', 1.0),  # dt 피처가 있으면 사용
            'success': row['result_name'] == 'Successful'
        }
        pass_events.append(event)

    return pass_events


if __name__ == "__main__":
    print("Ghost Defender Reconstructor Test")
    print("=" * 50)

    # 테스트 시나리오: 3개의 패스 시퀀스
    pass_sequence = [
        {
            'start_x': 35.0, 'start_y': 34.0,
            'end_x': 50.0, 'end_y': 40.0,
            'time_delta': 2.0,
            'success': True  # 성공
        },
        {
            'start_x': 50.0, 'start_y': 40.0,
            'end_x': 70.0, 'end_y': 30.0,
            'time_delta': 1.5,
            'success': True  # 성공
        },
        {
            'start_x': 70.0, 'start_y': 30.0,
            'end_x': 85.0, 'end_y': 45.0,
            'time_delta': 1.0,
            'success': False  # 실패 (인터셉트됨)
        },
    ]

    reconstructor = GhostDefenderReconstructor(num_defenders=4)

    print("\n패스 시퀀스:")
    for i, p in enumerate(pass_sequence):
        result = "성공" if p['success'] else "실패"
        print(f"  {i+1}. ({p['start_x']}, {p['start_y']}) -> ({p['end_x']}, {p['end_y']}) [{result}]")

    print("\n수비수 위치 복원 중...")
    final_coords = reconstructor.reconstruct_from_sequence(
        pass_sequence,
        n_iterations=100,
        verbose=True
    )

    print(f"\n복원된 수비수 좌표:")
    for i, coord in enumerate(final_coords):
        print(f"  수비수 {i+1}: ({coord[0].item():.1f}, {coord[1].item():.1f})")

    # 마지막 패스에 대한 인터셉트 확률 확인
    last_pass = pass_sequence[-1]
    ball_s = torch.tensor([last_pass['start_x'], last_pass['start_y']])
    ball_e = torch.tensor([last_pass['end_x'], last_pass['end_y']])

    with torch.no_grad():
        prob = reconstructor.physics(ball_s, ball_e, final_coords)

    print(f"\n마지막 패스 인터셉트 확률: {prob.item():.4f}")
    print(f"실제 결과: {'인터셉트(실패)' if not last_pass['success'] else '성공'}")
