# Zone-wise SHAP Analysis Report

## 개요
XGBoost 모델의 SHAP values를 활용하여 영역별(In-field vs Goal-line) 피처 중요도 패턴 분석

- **분석 대상**: end_x 예측 모델
- **샘플 수**: 3,087 (Validation set)
- **피처 수**: 221개

## 영역별 샘플 분포

| Zone | 샘플 수 | 비율 |
|------|---------|------|
| In-field | 9,298 | 60.2% |
| Top-out | 2,619 | 17.0% |
| Bottom-out | 2,675 | 17.3% |
| Goal-line | 843 | 5.5% |

---

## 1. Top 15 Mean |SHAP| per Zone

### In-field
| Rank | Feature | SHAP |
|------|---------|------|
| 1 | dist_to_goal_7 | +1.6679 |
| 2 | log_dist_to_goal_7 | +0.6399 |
| 3 | res_id_7 | -0.2913 |
| 4 | end_x_6 | +0.1719 |
| 5 | type_id_6 | +0.1170 |
| 6 | x_zone_7 | +0.0681 |
| 7 | last_result_encoded | -0.0574 |
| 8 | end_y_6 | +0.0540 |
| 9 | pressure_x_weight_7 | +0.0501 |
| 10 | start_x_6 | +0.0487 |

### Goal-line
| Rank | Feature | SHAP |
|------|---------|------|
| 1 | dist_to_goal_7 | +3.5154 |
| 2 | log_dist_to_goal_7 | +1.4901 |
| 3 | start_x_7 | +0.8436 |
| 4 | res_id_7 | +0.6793 |
| 5 | end_x_6 | +0.6594 |
| 6 | dt_7 | +0.4025 |
| 7 | type_id_6 | +0.3135 |
| 8 | dx_6 | -0.2339 |
| 9 | pressure_x_weight_7 | +0.2074 |
| 10 | last_result_encoded | +0.1215 |

---

## 2. Sign Reversal Analysis

**영역에 따라 SHAP 부호가 반전되는 피처 (6개 발견)**

| Feature | In-field | Goal-line | 해석 |
|---------|----------|-----------|------|
| **res_id_7** | -0.2913 | +0.6793 | 패스 실패 시: In-field에서는 end_x↓, Goal-line에서는 end_x↑ |
| **last_result_encoded** | -0.0574 | +0.1215 | 위와 동일한 패턴 |
| **angle_visible_7** | -0.0359 | +0.0407 | 골대 시야각의 영향 방향 반전 |
| **recent_lateral_abs_sum** | -0.0164 | +0.0470 | 횡방향 움직임의 영향 반전 |
| **recent_angle_std** | +0.0218 | -0.0226 | 방향 변화의 영향 반전 |
| **dist_6** | -0.0117 | +0.0241 | 이전 이동 거리의 영향 반전 |

### 핵심 인사이트: res_id_7 Sign Reversal
```
In-field:  Unsuccessful → end_x 감소 (수비에게 인터셉트, 공이 전진 못함)
Goal-line: Unsuccessful → end_x 증가 (공이 골라인 밖으로 나감)
```

---

## 3. Rank Shift Analysis (In-field vs Goal-line)

**순위 변화가 큰 Top 10 피처**

| Feature | IF Rank | GL Rank | Rank Δ |
|---------|---------|---------|--------|
| res_id_5 | 201 | 27 | +174 |
| action_angle_4 | 23 | 192 | -169 |
| last_pressure | 156 | 18 | +138 |
| dt_6 | 29 | 149 | -120 |
| angle_visible_5 | 128 | 19 | +109 |
| angle_to_goal_4 | 125 | 28 | +97 |
| dt_7 | 74 | 6 | +68 |
| start_y_7 | 26 | 92 | -66 |
| angle_visible_3 | 25 | 87 | -62 |
| action_angle_6 | 82 | 23 | +59 |

---

## 4. Magnitude Explosion Analysis

### Goal-line에서 영향력 증폭 Top 15 (ratio > 1)

| Feature | In-field | Goal-line | Ratio |
|---------|----------|-----------|-------|
| **dt_7** | 0.0067 | 0.4025 | **60.20x** |
| **last_pressure** | 0.0012 | 0.0459 | **38.37x** |
| **start_x_7** | 0.0302 | 0.8436 | **27.89x** |
| angle_visible_5 | 0.0024 | 0.0452 | 19.12x |
| is_zone14_7 | 0.0015 | 0.0221 | 14.50x |
| angle_to_goal_4 | 0.0025 | 0.0320 | 12.91x |
| end_y_4 | 0.0012 | 0.0136 | 11.51x |
| angle_visible_6 | 0.0070 | 0.0789 | 11.30x |
| dx_6 | 0.0237 | 0.2339 | 9.88x |
| prev_action_angle | 0.0018 | 0.0163 | 9.27x |

### Goal-line에서 영향력 감소 Top 10 (ratio < 1)

| Feature | In-field | Goal-line | Ratio |
|---------|----------|-----------|-------|
| dist_to_goal_0 | 0.0082 | 0.0001 | 0.01x |
| action_angle_4 | 0.0207 | 0.0003 | 0.01x |
| dist_to_goal_1 | 0.0094 | 0.0002 | 0.02x |
| polar_angle_2 | 0.0026 | 0.0001 | 0.03x |
| end_x_2 | 0.0080 | 0.0003 | 0.04x |

---

## 5. Winning Signals 판정

### Signal 1: Rank Shift (순위 ≥10 변화)
- **결과**: 32개 피처
- **판정**: ✅ 통과
- **의미**: 영역별 피처 중요도 구조가 확연히 다름
- **시사점**: FiLM/MoE 등 영역 조건부 모델링 근거 확보

### Signal 2: Sign Reversal
- **결과**: 6개 피처
- **판정**: ✅ 통과
- **의미**: 영역에 따라 피처 영향 방향이 반대
- **시사점**: 단일 모델로는 이 패턴을 학습하기 어려움, Zone-specific Head 또는 MoE 필요

### Signal 3: Magnitude Explosion (2x 이상 증폭)
- **결과**: 60개 피처
- **판정**: ✅ 통과
- **의미**: 경계 영역에서 특정 피처 영향력 폭증
- **시사점**: 영역별 전문화된 Head 필요

---

## 6. 결론 및 권장 모델 아키텍처

### 분석 결과 요약
| Winning Signal | 결과 | 판정 |
|----------------|------|------|
| Rank Shift (≥10) | 32개 | ✅ |
| Sign Reversal | 6개 | ✅ |
| Magnitude Explosion (≥2x) | 60개 | ✅ |

### 권장 아키텍처

1. **FiLM (Feature-wise Linear Modulation)**
   - Zone embedding으로 피처별 scale/shift 조절
   - Sign Reversal 패턴 학습 가능

2. **Mixture of Experts (MoE)**
   - Zone별 전문가 네트워크
   - Gating network가 zone에 따라 전문가 선택

3. **Zone-specific Heads**
   - 공유 backbone + 영역별 prediction head
   - Magnitude Explosion 대응

### 피처 설명 (dt_7 관련)

`dt_7`은 Data Leakage가 **아님**:
```
dt = df.groupby('game_episode')['time_seconds'].diff()
```
- `dt_7` = 이벤트 7의 time_seconds - 이벤트 6의 time_seconds
- 이전 이벤트 종료 후 현재 패스가 **시작**되기까지의 시간
- 패스 시작 시점은 알고 있음 (예측 대상은 end_x, end_y)

---

*Generated: 2025-12-29*
