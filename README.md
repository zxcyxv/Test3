# K-League Final Pass Prediction

K리그 실제 경기 이벤트 데이터를 기반으로, 주어진 플레이 시퀀스의 **마지막 패스 도착 좌표 `(X, Y)`**를 예측한 대회 참가 프로젝트입니다. 좌표는 경기장 차이를 보정하기 위해 FIFA 권장 규격인 `105 x 68` 그리드에 매핑된 상대 좌표를 사용합니다.

## 대회 성과

- 대회: K리그 경기 내 최종 패스 좌표 예측 AI 경진대회
- 주최/주관: 서울시립대, 한국프로축구연맹
- 운영: 데이콘
- 최종 순위: **937명 중 75위**
- 상위 비율: **상위 8.0%**

## 문제 정의

기존 축구 이벤트 데이터는 “누가 누구에게 패스했는가” 같은 단편적 사실을 기록하는 데 그치기 쉽습니다. 하지만 실제 경기에서 중요한 것은 선수 배치, 상대 압박, 공격 전개 방향, 터치라인/골라인과의 거리 같은 복합적인 맥락 속에서 **왜 특정 공간으로 패스가 향했는지**를 이해하는 것입니다.

이 프로젝트는 한 에피소드의 이벤트 시퀀스를 입력으로 받아 마지막 패스가 도착할 좌표를 예측합니다. 단순 좌표 회귀가 아니라, 공간적 경계와 전술적 맥락을 모델이 분리해서 보도록 설계했습니다.

## 핵심 접근

최종 파이프라인은 **FiLM + Spatial MoE Transformer가 만든 router signal**을 **XGBoost 회귀 모델의 입력 피처**로 결합합니다.

- Transformer는 이벤트 시퀀스 전체를 보고 공간적 라우팅 신호를 만듭니다.
- XGBoost는 K=8 wide feature와 router signal을 함께 사용해 최종 `end_x`, `end_y`를 회귀합니다.
- 검증은 5-fold OOF로 수행했고, 기록된 OOF distance는 약 `13.1m`입니다.

핵심은 Transformer 예측값 자체를 최종 답으로 쓰는 것이 아니라, **CLS token 기반 auxiliary head와 spatial router가 만든 `router_gate`, `router_zone_logit_0..3`를 XGBoost feature로 재사용**한 점입니다.

## 파이프라인

```text
raw event logs
  -> feature_engineering_k5.py
  -> train_features_k8.csv / test_features_k8.csv

train_features_v2.csv
  -> train_film_smoe2.py
  -> film_smoe_transformer.pt
  -> scripts/extract_router_features.py
  -> xgb_hybrid_features.csv

K=8 wide features + router features
  -> scripts/train_xgb_regressor.py
  -> xgb_full_x.json / xgb_full_y.json
  -> scripts/predict_xgb_submission.py
  -> submission_xgb_hybrid.csv
```

## 주요 파일

- `feature_engineering_k5.py`: 마지막 `K=8`개 이벤트를 wide tabular feature로 변환합니다. 마지막 이벤트의 `end_x`, `end_y` 관련 누수 피처는 제외합니다.
- `train_film_smoe2.py`: FiLM, Spatial MoE, CLS token, boundary auxiliary task를 포함한 Transformer router를 학습합니다.
- `scripts/extract_router_features.py`: 학습된 Transformer에서 `router_gate`와 4-class boundary logits를 추출합니다.
- `scripts/train_xgb_regressor.py`: K=8 wide feature와 router feature를 merge해 XGBoost `end_x`, `end_y` 회귀 모델을 학습합니다.
- `scripts/predict_xgb_submission.py`: 저장된 XGBoost 모델로 제출 파일을 생성합니다.
- `scripts/diagnose_router.py`: router gate가 공간 경계를 제대로 분리하는지 시각적으로 점검하는 보조 진단 도구입니다.

## 모델링 포인트

- **CLS token**: 이벤트 시퀀스 전체를 요약하는 global context로 사용합니다.
- **FiLM conditioning**: 마지막 위치 `(start_x_7, start_y_7)`로 Transformer representation을 위치 조건부로 보정합니다.
- **Spatial MoE router**: in-field expert와 boundary expert를 나누기 위한 gate를 학습합니다.
- **Auxiliary boundary logits**: In-field / Top-out / Bottom-out / Goal-line의 4-class logits를 만들고, 이를 XGBoost 입력으로 넘깁니다.
- **XGBoost final regressor**: 경계 근처의 비선형 분기와 tabular interaction을 처리합니다.

## Transformer Forward 구조

`train_film_smoe2.py`의 `FiLMSpatialMoETransformer.forward()`는 다음 순서로 동작합니다.

```text
Input sequence X
  shape: [batch, 8, 24]
  last event position: (start_x_7, start_y_7)

        ┌─────────────────────────────────────────┐
        │ 1. 마지막 위치 추출                      │
        │    pos = X[:, -1, (start_x, start_y)]   │
        └────────────────────┬────────────────────┘
                             │
                             ├──────────────┐
                             │              │
                             ▼              ▼
        ┌───────────────────────┐     ┌─────────────────────────┐
        │ SpatialConditioningNet │     │ SpatialRouter           │
        │ pos -> gamma, beta     │     │ pos + boundary distance │
        │ Fourier(pos) + MLP     │     │ + Fourier -> gate       │
        └───────────┬───────────┘     └────────────┬────────────┘
                    │                              │
                    ▼                              │
┌──────────────────────────────────────┐           │
│ 2. 좌표 Fourier feature 생성          │           │
│    [start_x, start_y, end_x, end_y]   │           │
│    -> cos/sin Fourier features        │           │
└──────────────────┬───────────────────┘           │
                   ▼                               │
┌──────────────────────────────────────┐           │
│ 3. Fourier feature + 원본 feature concat          │
│    Linear projection -> d_model=128   │           │
└──────────────────┬───────────────────┘           │
                   ▼                               │
┌──────────────────────────────────────┐           │
│ 4. CLS token 추가 + position embedding │          │
└──────────────────┬───────────────────┘           │
                   ▼                               │
┌──────────────────────────────────────┐           │
│ 5. Gated Transformer Block x 2        │           │
│    MultiheadAttention output을 gate로 │           │
│    조절한 뒤 residual 연결            │           │
└──────────────────┬───────────────────┘           │
                   ▼                               │
┌──────────────────────────────────────┐           │
│ 6. 각 Transformer layer 뒤 FiLM 적용  │           │
│    h = gamma * RMSNorm(h) + beta      │           │
└──────────────────┬───────────────────┘           │
                   ▼                               │
┌──────────────────────────────────────┐           │
│ 7. final RMSNorm 후 CLS vector 추출   │           │
│    cls_out = x[:, 0, :]               │           │
└──────────────────┬───────────────────┘           │
                   │                               │
                   ├──────────────┬────────────────┘
                   ▼              ▼
┌────────────────────────┐   ┌────────────────────────┐
│ In-field Expert         │   │ Boundary Expert         │
│ cls -> mu_A, logvar_A,  │   │ cls -> mu_B, logvar_B,  │
│        residual_A       │   │        residual_B       │
└───────────┬────────────┘   └───────────┬────────────┘
            └──────────────┬─────────────┘
                           ▼
        ┌─────────────────────────────────────────┐
        │ 8. gate로 두 expert 출력 혼합            │
        │ mu      = gate * mu_B + (1-gate) * mu_A │
        │ logvar  = gate * lv_B + (1-gate) * lv_A │
        │ residual= gate * r_B  + (1-gate) * r_A  │
        └────────────────────┬────────────────────┘
                             ▼
        ┌─────────────────────────────────────────┐
        │ 9. Heteroscedastic output               │
        │ sigma = exp(0.5 * logvar)               │
        │ y_final = mu + sigma * residual         │
        └─────────────────────────────────────────┘

        ┌─────────────────────────────────────────┐
        │ 10. Auxiliary head from CLS             │
        │ aux_dist: boundary distance regression  │
        │ aux_zone: 4-class boundary logits       │
        └─────────────────────────────────────────┘
```

XGBoost로 넘어가는 값은 Transformer의 최종 좌표 예측이 아니라 `SpatialRouter`의 `gate`와 auxiliary head의 `aux_zone` logits입니다. 이 값들이 `router_gate`, `router_zone_logit_0..3` 컬럼으로 저장되어 K=8 wide feature와 merge됩니다.

## Loss 설계

Transformer router 학습은 단일 MSE가 아니라 multi-task loss로 구성했습니다.

```text
total =
  w_nll      * GaussianNLL(y, mu, log_var)
+ w_final    * MSE(y_final, y)
+ w_aux_dist * SmoothL1(aux_dist, boundary_distance)
+ w_aux_zone * CrossEntropy(aux_zone, boundary_zone, label_smoothing=0.1)
+ w_gate     * BCE(gate, is_boundary)
```

특별히 중요한 부분은 `gate` supervision입니다. `boundary_zone == 0`인 in-field 샘플은 gate target을 `0`, `boundary_zone in {1,2,3}`인 경계 샘플은 `1`로 두고 BCE를 걸어 router collapse를 막았습니다.

학습 중에는 router temperature도 스케줄링합니다.

- Warm-up: `T = 2.0`, soft routing, gate loss 가중치 `2.0`
- Anneal: `T = 2.0 -> 0.5`, auxiliary/gate 가중치 점진 조정
- Fine-tune: `T = 0.5`, sharper routing
- Validation/router feature 추출: `T = 0.05`로 더 hard한 routing signal 사용

## 데이터 구성

대회 데이터는 다음 구조를 가정합니다.

- `train.csv`: 학습 이벤트 데이터
- `test.csv`: 평가 대상 episode 경로
- `test/`: `{game_id}/{game_id}_{episode}.csv` 형식의 평가 episode 파일
- `match_info.csv`: 경기 메타데이터
- `sample_submission.csv`: 제출 양식
- `data_description.xlsx`: 데이터 명세

코드는 `open_track1/` 아래에 위 파일들이 있다고 가정합니다. 원본 데이터, 학습 체크포인트, XGBoost 모델 JSON, 제출 CSV는 저장소에 포함하지 않습니다.

## 실행 방법

의존성은 `uv` 기준입니다.

```bash
uv sync
```

K=8 wide feature를 생성합니다.

```bash
uv run python feature_engineering_k5.py
```

Transformer router를 학습합니다.

```bash
uv run python train_film_smoe2.py
```

학습된 router에서 XGBoost용 feature를 추출합니다.

```bash
uv run python scripts/extract_router_features.py
uv run python scripts/extract_router_features_test.py
```

XGBoost 회귀 모델을 학습하고 제출 파일을 생성합니다.

```bash
uv run python scripts/train_xgb_regressor.py
uv run python scripts/predict_xgb_submission.py
```

## 기술 스택

- Python, uv
- PyTorch
- XGBoost
- pandas, NumPy
- scikit-learn
- SciPy, matplotlib

## 참고 문서

- [XGBoost Hybrid Pipeline](XGB_HYBRID_PIPELINE.md)
- [XGBoost Feature Specification](XGB_FEATURES.md)
- [FiLM + Spatial MoE Transformer](TRAIN_FILM_SMOE2.md)
