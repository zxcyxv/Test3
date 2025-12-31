# Ghost Defender Reconstructor Algorithm

## 1. Overview

### 1.1 Goal
**패스 성공/실패 결과만을 이용하여 수비수의 위치를 역추론(Reverse Engineering)하는 것**

축구 경기 데이터에서 우리가 가진 정보:
- 패스 시작 위치 (start_x, start_y)
- 패스 종료 위치 (end_x, end_y)
- 패스 결과 (Successful / Unsuccessful)
- 패스 소요 시간 (time_delta)

우리가 **없는** 정보:
- 수비수 위치 (실제 트래킹 데이터 없음)
- 공격수 위치
- 선수 속도, 방향

### 1.2 Why?
수비수 위치를 알면:
- 패스 성공 확률 예측 가능
- 최적의 패스 목표 지점 추론 가능
- 마지막 패스의 end_x, end_y 예측에 활용

---

## 2. Theoretical Background

### 2.1 Spearman's Pitch Control Model
**출처**: "Beyond Expected Goals" (MIT Sloan Sports Analytics Conference, 2018)

핵심 원리: **"누가 먼저 공에 도착하느냐"** (Time-to-Intercept)

```
P(intercept) = σ(-λ × (T_ball - T_defender))
```

Where:
- `T_ball`: 공이 목표 지점에 도착하는 시간
- `T_defender`: 가장 가까운 수비수가 목표 지점에 도착하는 시간
- `λ`: 민감도 계수 (Spearman 권장값: 4.3)
- `σ`: 시그모이드 함수

### 2.2 Time Calculation

**공의 이동 시간:**
```
T_ball = distance(start, end) / ball_speed
```
- 기본 공 속도: 15 m/s
- 실제 데이터가 있으면 time_delta 사용

**수비수 도착 시간:**
```
T_defender = T_reaction + distance(defender, target) / V_max
```
- 반응 시간 (T_reaction): 0.5초
- 최대 속도 (V_max): 7.5 m/s

### 2.3 Intercept Probability Interpretation

| T_ball vs T_defender | 의미 | P(intercept) |
|---------------------|------|--------------|
| T_ball >> T_defender | 수비가 먼저 도착 | ≈ 1 (높음) |
| T_ball ≈ T_defender | 경쟁 상황 | ≈ 0.5 |
| T_ball << T_defender | 공이 먼저 통과 | ≈ 0 (낮음) |

---

## 3. Algorithm Detail

### 3.1 Overall Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Episode Data                       │
│  - K-1 passes before final pass                             │
│  - Each pass: (start, end, success, time_delta)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 1: Initialize Defender Positions           │
│  - 4 defenders                                               │
│  - x: random(50, 90)  ← Always right side                   │
│  - y: random(14, 54)  ← Middle vertical band                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Step 2: Optimization Loop (30 iterations)       │
│                                                              │
│  For each iteration:                                         │
│    total_loss = 0                                            │
│                                                              │
│    For each pass in episode:                                 │
│      1. Calculate intercept probability                      │
│      2. Calculate Reality Loss                               │
│      3. Calculate Tactic Loss                                │
│      4. Calculate Boundary Loss                              │
│      total_loss += loss                                      │
│                                                              │
│    Backpropagate & Update defender coordinates               │
│    Clip coordinates to field boundaries                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Output: Reconstructed Defender Positions        │
│  - 4 × (x, y) coordinates                                    │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Loss Functions

#### 3.2.1 Reality Loss (Main Loss)
패스 결과와 예측 확률의 일치도

```python
prob_intercept = physics_engine(ball_start, ball_end, defender_coords)

if pass_success:
    # 패스 성공 → 인터셉트 확률이 낮아야 함
    loss_reality = prob_intercept ** 2
else:
    # 패스 실패 → 인터셉트 확률이 높아야 함
    loss_reality = (1 - prob_intercept) ** 2
```

**직관적 해석:**
- 패스 성공 시: 수비수가 패스 경로에서 멀리 있어야 함 → 수비수를 멀리 밀어냄
- 패스 실패 시: 수비수가 패스 경로에 가까이 있어야 함 → 수비수를 경로 쪽으로 당김

#### 3.2.2 Tactic Loss (Regularization)
수비수가 전술적으로 합리적인 위치에 있도록 제약

```python
# 수비수 무게중심
defender_center = mean(defender_coords)

# 이상적 위치: 공과 골대 사이
ideal_position = (ball_start + goal_position) / 2

# L2 거리에 가중치 적용
loss_tactic = ||defender_center - ideal_position|| × 0.05
```

**문제점:** 가중치 0.05가 너무 작아서 거의 영향 없음

#### 3.2.3 Boundary Loss (Soft Constraint)
수비수가 경기장 밖으로 나가지 않도록

```python
out_x = ReLU(-x) + ReLU(x - 105)  # x가 0 미만이거나 105 초과 시 패널티
out_y = ReLU(-y) + ReLU(y - 68)   # y가 0 미만이거나 68 초과 시 패널티
loss_boundary = (sum(out_x) + sum(out_y)) × 0.1
```

### 3.3 Optimization

```python
optimizer = Adam(defender_coords, lr=1.0)

for iteration in range(30):
    optimizer.zero_grad()

    total_loss = sum of losses for all passes

    total_loss.backward()
    optimizer.step()

    # Hard constraint: clip to field
    defender_coords[:, 0].clamp_(0, 105)
    defender_coords[:, 1].clamp_(0, 68)
```

---

## 4. Physics Engine Detail

### 4.1 PhysicsPitchControl Class

```python
class PhysicsPitchControl:
    def __init__(self):
        self.pitch_length = 105.0      # 필드 길이 (m)
        self.pitch_width = 68.0        # 필드 너비 (m)
        self.max_player_speed = 7.5    # 수비수 최대 속도 (m/s)
        self.reaction_time = 0.5       # 반응 시간 (초)
        self.lambda_coef = 4.3         # 시그모이드 민감도
```

### 4.2 Forward Pass

```python
def forward(ball_start, ball_end, defender_positions, time_delta):
    # 1. 공의 이동 시간
    if time_delta is not None:
        ball_time = time_delta
    else:
        ball_time = distance(ball_start, ball_end) / 15.0

    # 2. 수비수들의 도착 시간
    for each defender:
        dist = distance(defender, ball_end)
        time = 0.5 + dist / 7.5

    # 3. 가장 빨리 도착하는 수비수
    min_defender_time = min(defender_times)

    # 4. 인터셉트 확률
    time_diff = ball_time - min_defender_time
    prob = sigmoid(-4.3 × time_diff)

    return prob
```

### 4.3 Numerical Example

**상황:**
- 공: (50, 34) → (80, 40)
- 거리: 31.6m
- 공 이동 시간: 31.6 / 15 = 2.1초

**수비수 A가 (70, 35)에 있을 때:**
- 거리: sqrt((80-70)² + (40-35)²) = 11.2m
- 도착 시간: 0.5 + 11.2/7.5 = 1.99초
- time_diff = 2.1 - 1.99 = 0.11
- P(intercept) = sigmoid(-4.3 × 0.11) = 0.38

**수비수 B가 (78, 39)에 있을 때:**
- 거리: sqrt((80-78)² + (40-39)²) = 2.2m
- 도착 시간: 0.5 + 2.2/7.5 = 0.79초
- time_diff = 2.1 - 0.79 = 1.31
- P(intercept) = sigmoid(-4.3 × 1.31) = 0.003

---

## 5. Critical Problems

### 5.1 Initialization Bias

```python
def _reset_coords(self):
    self.coords[:, 0] = torch.rand(4) * 40 + 50  # x: 50~90
    self.coords[:, 1] = torch.rand(4) * 40 + 14  # y: 14~54
```

**문제:**
- 수비수가 항상 x=50~90 영역에서 시작
- 공이 x=30에 있어도 수비수는 x=50~90에서 시작
- 최적화가 충분히 강하지 않아 초기 위치에서 크게 벗어나지 못함

### 5.2 Information Insufficiency

| 복원 대상 | 가용 정보 |
|----------|----------|
| 4명 × 2좌표 = **8개 변수** | 2~3개 패스 × 1비트(성공/실패) = **2~3비트** |

**근본적 문제:** 8개의 연속 변수를 2~3비트의 이산 정보로 복원하는 것은 수학적으로 불가능

### 5.3 Weak Tactic Constraint

```python
loss_tactic = ||defender_center - ideal_position|| × 0.05
```

- 가중치 0.05가 너무 작음
- Reality Loss에 비해 영향력이 미미
- 수비수가 전술적으로 합리적인 위치로 이동하도록 강제하지 못함

### 5.4 Local Minima

- 30회 반복은 글로벌 최적해를 찾기에 부족
- Adam optimizer의 learning rate 1.0이 적절한지 불명확
- 여러 로컬 미니마 중 하나에 수렴할 가능성 높음

---

## 6. Experimental Evidence

### 6.1 Defender Position Distribution

실제 복원 결과 분석:

| 통계 | defender_x | defender_y |
|------|-----------|-----------|
| Mean | 75~78m | 33~35m |
| Std | 15~17m | 13~17m |
| Min | 20~30m | 0m |
| Max | 105m | 68m |

**관찰:** 공의 위치와 관계없이 수비수들이 항상 x=75 근처에 몰려 있음

### 6.2 Model Performance Impact

| Model | Mean Euclidean Distance |
|-------|------------------------|
| 기존 피처 (K=8) | 14.6295 m |
| + 물리 피처 | 14.6915 m (+0.06m) |

**결론:** 물리 피처가 성능 향상에 기여하지 못함 → 복원된 수비수 좌표가 의미 없음

---

## 7. File Structure

```
ReverseSoccerMap/
├── physics_engine.py       # Spearman's Pitch Control 구현
├── ghost_reconstructor.py  # 수비수 위치 역추론 알고리즘
├── feature_generator.py    # 복원된 좌표로 피처 생성
├── main.py                 # 파이프라인 실행
├── compare_with_physics.py # 성능 비교 실험
└── ALGORITHM.md            # 이 문서
```

---

## 8. Potential Improvements

### 8.1 Dynamic Initialization
공의 위치에 따라 초기 수비수 위치 조정:
```python
def _reset_coords(self, ball_position):
    # 공과 골대 사이에 수비수 초기화
    center_x = (ball_position[0] + 105) / 2
    self.coords[:, 0] = torch.rand(4) * 20 + center_x - 10
```

### 8.2 Stronger Tactic Constraint
```python
loss_tactic = ||defender_center - ideal_position|| × 0.5  # 0.05 → 0.5
```

### 8.3 More Iterations & Better Optimizer
```python
n_iterations = 100  # 30 → 100
optimizer = LBFGS(...)  # Adam → LBFGS for better convergence
```

### 8.4 Ensemble Approach
여러 번 복원하여 평균:
```python
all_coords = [reconstruct() for _ in range(10)]
final_coords = mean(all_coords)
```

### 8.5 Alternative Approach
수비수 좌표 직접 복원 대신:
- 수비 밀집도만 추정
- 패스 성공 확률 직접 모델링
- 영역별 위험도 계산

---

## 9. Conclusion

현재 Ghost Defender Reconstructor는 이론적으로 흥미로운 접근이지만, 실용적 한계가 명확함:

1. **정보 부족**: 2~3비트로 8개 변수 복원 불가
2. **초기화 편향**: 항상 오른쪽에서 시작
3. **약한 제약**: 전술적 합리성 강제 부족
4. **검증 불가**: 실제 수비수 위치 데이터 없음

**권장사항:** 수비수 좌표 직접 복원보다는, 패스 성공/실패 패턴을 직접 모델링하는 것이 더 효과적일 수 있음.
