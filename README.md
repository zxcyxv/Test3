# Soccer Router XGBoost

K리그 이벤트 데이터로 에피소드 마지막 패스의 도착 좌표(`end_x`, `end_y`)를 예측하는 하이브리드 모델링 프로젝트입니다. 최종 파이프라인은 **FiLM + Spatial MoE Transformer가 만든 router signal**을 **XGBoost 회귀 모델의 입력 피처**로 결합합니다.

## 핵심 성과

- 대표 파이프라인: `FiLMSpatialMoETransformer` router signal + K=8 wide features + XGBoost
- 검증 방식: 5-fold OOF
- 기록된 성능: 약 `13.1m` OOF distance
- 예측 대상: 마지막 이벤트의 패스 도착 좌표 `(end_x, end_y)`

## 왜 이 구조를 썼나

패스 도착 좌표 예측은 일반적인 회귀 문제처럼 보이지만, 실제 오차는 경계 영역에서 크게 튑니다. 중앙 지역의 일반 패스와 터치라인/골라인 근처의 패스는 다른 규칙을 따르기 때문에, 하나의 smooth regressor만으로는 경계 판단이 둔해집니다.

이 프로젝트는 이 문제를 두 단계로 나눴습니다.

1. Transformer가 시퀀스 맥락과 마지막 위치를 보고 공간적 라우팅 신호를 만듭니다.
2. XGBoost가 원본 wide feature와 router signal을 함께 사용해 최종 좌표를 회귀합니다.

핵심은 Transformer의 전체 예측값을 그대로 쓰는 것이 아니라, **CLS token 기반 auxiliary head와 spatial router가 만든 `router_gate`, `router_zone_logit_0..3`를 XGBoost feature로 재사용**한다는 점입니다.

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

## 주요 구성

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

## 데이터/산출물

코드는 `open_track1/` 아래 대회 데이터를 기대합니다. 대용량 원본 데이터, 학습 체크포인트, XGBoost 모델 JSON, 제출 CSV는 저장소에 포함하지 않는 것을 권장합니다.

예상 산출물:

- `open_track1/train_features_k8.csv`
- `open_track1/test_features_k8.csv`
- `open_track1/film_smoe_transformer.pt`
- `open_track1/xgb_hybrid_features.csv`
- `open_track1/xgb_hybrid_features_test.csv`
- `models/xgb_full_x.json`
- `models/xgb_full_y.json`
- `open_track1/submission_xgb_hybrid.csv`

## 참고 문서

- [XGBoost Hybrid Pipeline](XGB_HYBRID_PIPELINE.md)
- [XGBoost Feature Specification](XGB_FEATURES.md)
- [FiLM + Spatial MoE Transformer](TRAIN_FILM_SMOE2.md)
