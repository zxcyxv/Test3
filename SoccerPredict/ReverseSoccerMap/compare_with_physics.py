"""
Physics Features + XGBoost 성능 비교
GPU 가속 사용
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

from physics_engine import PhysicsPitchControl
from ghost_reconstructor import GhostDefenderReconstructor
from feature_generator import PhysicsFeatureGenerator

DATA_DIR = Path('/workspace/SoccerPredict/open_track1')
FIELD_X, FIELD_Y = 105, 68

# GPU 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


class GPUGhostDefenderReconstructor(torch.nn.Module):
    """GPU 가속 수비수 복원기"""

    def __init__(self, num_defenders=4, device='cuda'):
        super().__init__()
        self.num_defenders = num_defenders
        self.device = device
        self.pitch_length = 105.0
        self.pitch_width = 68.0
        self.max_player_speed = 7.5
        self.reaction_time = 0.5
        self.lambda_coef = 4.3

        # 수비수 좌표 초기화
        init_coords = torch.zeros(num_defenders, 2, device=device)
        init_coords[:, 0] = torch.rand(num_defenders, device=device) * 40 + 50
        init_coords[:, 1] = torch.rand(num_defenders, device=device) * 40 + 14
        self.coords = torch.nn.Parameter(init_coords)

        self.goal_pos = torch.tensor([105.0, 34.0], device=device)

    def forward(self, ball_start, ball_end, time_delta=None):
        """인터셉트 확률 계산"""
        # 공 이동 시간
        if time_delta is not None:
            ball_time = time_delta
        else:
            dist = torch.norm(ball_end - ball_start)
            ball_time = dist / 15.0

        # 수비수 도착 시간
        defender_dists = torch.norm(self.coords - ball_end, dim=-1)
        defender_times = self.reaction_time + (defender_dists / self.max_player_speed)
        min_defender_time = torch.min(defender_times)

        # 인터셉트 확률
        time_diff = ball_time - min_defender_time
        prob = torch.sigmoid(self.lambda_coef * (-time_diff))

        return prob

    def compute_loss(self, ball_start, ball_end, pass_success, time_delta=None):
        prob = self.forward(ball_start, ball_end, time_delta)

        if pass_success:
            loss_reality = prob ** 2
        else:
            loss_reality = (1 - prob) ** 2

        # Tactic loss
        def_center = torch.mean(self.coords, dim=0)
        ideal_pos = (ball_start + self.goal_pos) / 2
        loss_tactic = torch.norm(def_center - ideal_pos) * 0.05

        return loss_reality + loss_tactic

    def reconstruct(self, pass_events, n_iterations=30):
        """패스 시퀀스로부터 수비수 복원"""
        # 초기화
        with torch.no_grad():
            self.coords[:, 0] = torch.rand(self.num_defenders, device=self.device) * 40 + 50
            self.coords[:, 1] = torch.rand(self.num_defenders, device=self.device) * 40 + 14

        optimizer = torch.optim.Adam([self.coords], lr=1.0)

        for _ in range(n_iterations):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)

            for event in pass_events:
                ball_start = torch.tensor([event['start_x'], event['start_y']], device=self.device)
                ball_end = torch.tensor([event['end_x'], event['end_y']], device=self.device)
                time_delta = torch.tensor(event.get('time_delta', 1.0), device=self.device)

                loss = self.compute_loss(ball_start, ball_end, event['success'], time_delta)
                total_loss = total_loss + loss

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.coords[:, 0].clamp_(0, self.pitch_length)
                self.coords[:, 1].clamp_(0, self.pitch_width)

        return self.coords.detach().clone()


class GPUFeatureGenerator:
    """GPU 가속 피처 생성기"""

    def __init__(self, device='cuda'):
        self.device = device
        self.pitch_length = 105.0
        self.pitch_width = 68.0

    def compute_features(self, ball_start, defender_positions):
        """피처 계산"""
        features = {}

        # 거리 피처
        distances = torch.norm(defender_positions - ball_start, dim=-1)
        features['min_defender_dist'] = distances.min().item()
        features['mean_defender_dist'] = distances.mean().item()
        features['std_defender_dist'] = distances.std().item()

        # 밀집도
        features['defenders_within_5m'] = (distances < 5).sum().item()
        features['defenders_within_10m'] = (distances < 10).sum().item()

        # 골대 방향 원뿔
        goal_pos = torch.tensor([self.pitch_length, self.pitch_width / 2], device=self.device)
        goal_direction = goal_pos - ball_start
        goal_direction = goal_direction / torch.norm(goal_direction)

        defender_directions = defender_positions - ball_start
        defender_distances = torch.norm(defender_directions, dim=-1)
        cos_angles = torch.sum(defender_directions * goal_direction, dim=-1) / (defender_distances + 1e-6)

        features['defenders_in_goal_cone_30'] = (cos_angles > 0.866).sum().item()
        features['defenders_in_goal_cone_45'] = (cos_angles > 0.707).sum().item()

        # 수비 라인
        features['defense_line_x'] = defender_positions[:, 0].mean().item()
        features['defense_width'] = defender_positions[:, 1].std().item()

        # 경계 피처
        features['defenders_near_top'] = (defender_positions[:, 1] > 60).sum().item()
        features['defenders_near_bottom'] = (defender_positions[:, 1] < 8).sum().item()
        features['defenders_near_goal'] = (defender_positions[:, 0] > 100).sum().item()

        # 수비수 좌표
        for i, coord in enumerate(defender_positions):
            features[f'defender_{i}_x'] = coord[0].item()
            features[f'defender_{i}_y'] = coord[1].item()

        return features


def extract_episode_passes(episode_df):
    """에피소드에서 패스 추출"""
    df = episode_df.iloc[:-1].copy()
    passes = df[df['type_name'] == 'Pass']
    events = []
    prev_time = None

    for _, row in passes.iterrows():
        current_time = row['time_seconds']
        time_delta = current_time - prev_time if prev_time else 1.0
        event = {
            'start_x': float(row['start_x']),
            'start_y': float(row['start_y']),
            'end_x': float(row['end_x']),
            'end_y': float(row['end_y']),
            'time_delta': max(time_delta, 0.1),
            'success': row['result_name'] == 'Successful'
        }
        events.append(event)
        prev_time = current_time

    return events


def generate_physics_features_gpu(raw_df, episode_ids, device='cuda'):
    """GPU로 물리 피처 생성"""
    reconstructor = GPUGhostDefenderReconstructor(num_defenders=4, device=device)
    feature_gen = GPUFeatureGenerator(device=device)

    episodes = raw_df.groupby('game_episode')
    all_features = []

    for ep_id in tqdm(episode_ids, desc='Generating physics features (GPU)'):
        episode_df = episodes.get_group(ep_id).sort_values('action_id')

        try:
            pass_events = extract_episode_passes(episode_df)
            last_row = episode_df.iloc[-1]
            last_start = torch.tensor(
                [float(last_row['start_x']), float(last_row['start_y'])],
                device=device
            )

            if len(pass_events) >= 1:
                recent_events = pass_events[-3:] if len(pass_events) > 3 else pass_events
                defender_coords = reconstructor.reconstruct(recent_events, n_iterations=30)
            else:
                defender_coords = torch.tensor([
                    [60.0, 25.0], [70.0, 35.0], [65.0, 45.0], [75.0, 30.0]
                ], device=device)

            features = feature_gen.compute_features(last_start, defender_coords)
            features['game_episode'] = ep_id
            all_features.append(features)

        except Exception as e:
            continue

    return pd.DataFrame(all_features)


def create_boundary_labels(end_x, end_y):
    """경계 라벨 생성"""
    labels = np.zeros(len(end_x), dtype=int)
    labels[(end_x > FIELD_X - 5) | (end_x < 5)] = 3
    labels[(labels == 0) & (end_y > FIELD_Y - 5)] = 1
    labels[(labels == 0) & (end_y < 5)] = 2
    return labels


def main():
    print('=' * 60)
    print('Physics Features + XGBoost 성능 비교 (GPU)')
    print('=' * 60)

    # 1. 데이터 로드
    print('\n[1] 데이터 로드...')
    existing_df = pd.read_csv(DATA_DIR / 'train_features_k8.csv')
    raw_df = pd.read_csv(DATA_DIR / 'train.csv')
    print(f'기존 피처: {len(existing_df)} 에피소드')

    # 2. 물리 피처 생성 (전체 또는 샘플)
    print('\n[2] 물리 피처 생성...')
    episode_ids = existing_df['game_episode'].tolist()

    # 전체 생성 (GPU로 빠름)
    physics_df = generate_physics_features_gpu(raw_df, episode_ids, device=DEVICE)
    print(f'물리 피처: {len(physics_df)} 에피소드, {len(physics_df.columns)-1}개 피처')

    # 저장
    physics_df.to_csv(DATA_DIR / 'physics_features_full.csv', index=False)
    print(f'저장: {DATA_DIR / "physics_features_full.csv"}')

    # 3. 병합
    print('\n[3] 피처 병합...')
    merged_df = existing_df.merge(physics_df, on='game_episode', how='inner')
    print(f'병합: {len(merged_df)} 에피소드')

    # 4. 라벨
    y = create_boundary_labels(merged_df['target_end_x'].values, merged_df['target_end_y'].values)

    # 5. 피처 컬럼
    exclude = ['game_episode', 'target_end_x', 'target_end_y', 'last_result_name', 'last_result_encoded']
    existing_cols = [c for c in existing_df.columns if c not in exclude
                     and existing_df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
    physics_cols = [c for c in physics_df.columns if c != 'game_episode']
    all_cols = existing_cols + physics_cols

    X_existing = merged_df[existing_cols].fillna(0).values
    X_all = merged_df[all_cols].fillna(0).values

    print(f'기존 피처: {len(existing_cols)}개')
    print(f'물리 피처: {len(physics_cols)}개')
    print(f'전체 피처: {len(all_cols)}개')

    # 6. Train/Val 분리
    X_train_exist, X_val_exist, y_train, y_val = train_test_split(
        X_existing, y, test_size=0.2, random_state=42
    )
    X_train_all, X_val_all, _, _ = train_test_split(
        X_all, y, test_size=0.2, random_state=42
    )

    # 7. XGBoost 학습
    print('\n[4] XGBoost 학습...')
    model_cfg = dict(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=-1, tree_method='hist'
    )

    # 기존 피처만
    print('\n--- 기존 피처만 (K=8) ---')
    model_exist = XGBClassifier(**model_cfg)
    model_exist.fit(X_train_exist, y_train, eval_set=[(X_val_exist, y_val)], verbose=False)
    preds_exist = model_exist.predict(X_val_exist)
    acc_exist = accuracy_score(y_val, preds_exist)

    infield_mask = y_val == 0
    boundary_mask = y_val > 0
    ir_exist = (preds_exist[infield_mask] == 0).mean()
    br_exist = (preds_exist[boundary_mask] > 0).mean()

    print(f'Accuracy: {acc_exist:.4f}')
    print(f'In-field Recall: {ir_exist:.4f}')
    print(f'Boundary Recall: {br_exist:.4f}')

    # 기존 + 물리 피처
    print('\n--- 기존 + 물리 피처 ---')
    model_all = XGBClassifier(**model_cfg)
    model_all.fit(X_train_all, y_train, eval_set=[(X_val_all, y_val)], verbose=False)
    preds_all = model_all.predict(X_val_all)
    acc_all = accuracy_score(y_val, preds_all)

    ir_all = (preds_all[infield_mask] == 0).mean()
    br_all = (preds_all[boundary_mask] > 0).mean()

    print(f'Accuracy: {acc_all:.4f}')
    print(f'In-field Recall: {ir_all:.4f}')
    print(f'Boundary Recall: {br_all:.4f}')

    # 8. 결과 비교
    print('\n' + '=' * 60)
    print('결과 비교')
    print('=' * 60)
    print(f'| 설정              | Accuracy | In-field | Boundary | 차이     |')
    print(f'|-------------------|----------|----------|----------|----------|')
    print(f'| 기존 피처 (K=8)   | {acc_exist:.4f}   | {ir_exist:.4f}   | {br_exist:.4f}   | -        |')
    print(f'| + 물리 피처       | {acc_all:.4f}   | {ir_all:.4f}   | {br_all:.4f}   | {acc_all-acc_exist:+.4f}   |')

    # 9. 물리 피처 중요도
    print('\n물리 피처 중요도 Top 10:')
    importances = model_all.feature_importances_
    feat_imp = sorted(zip(all_cols, importances), key=lambda x: -x[1])
    physics_imp = [(f, i) for f, i in feat_imp if f in physics_cols]

    for rank, (feat, imp) in enumerate(physics_imp[:10], 1):
        print(f'  {rank}. {feat}: {imp:.4f}')

    return merged_df, model_all


if __name__ == '__main__':
    merged_df, model = main()
