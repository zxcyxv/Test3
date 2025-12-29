# Soccer Pass End Position Prediction Project

## Project Overview
축구 경기 이벤트 데이터를 분석하여 각 에피소드의 마지막 패스(Pass)의 종료 위치 (end_x, end_y)를 예측하는 대회 프로젝트

## Evaluation
- **Metric**: Euclidean Distance (예측 좌표와 실제 좌표 간의 유클리드 거리)
- **Field Size**: 105m (x) × 68m (y)

## Data Structure
```
open_track1/
├── train.csv              # 356,721 actions, 15,435 episodes, 198 games
├── test.csv               # 2,414 episodes (prediction target)
├── sample_submission.csv  # Submission format: game_episode, end_x, end_y
├── data_description.xlsx
└── test/                  # 30 game folders with individual episode CSVs
```

## Key Columns
| Column | Description |
|--------|-------------|
| game_episode | Unique episode identifier (e.g., 153363_1) |
| type_name | Action type: Pass, Carry, Duel, Recovery, Interception, etc. |
| result_name | Successful, Unsuccessful, NoResult |
| start_x, start_y | Action start coordinates (0~105, 0~68) |
| end_x, end_y | Action end coordinates (target for last Pass) |

## Key Insights
- 모든 에피소드의 마지막 액션은 항상 **Pass**
- 테스트 데이터의 마지막 Pass에서 end_x, end_y가 비어있음 (예측 대상)
- Successful vs Unsuccessful 패스의 end_x, end_y 분포가 크게 다름
- 이전 액션들의 시퀀스가 마지막 패스 위치 예측에 중요

## Commands
```bash
# Data location
/workspace/SoccerPredict/open_track1/
```
