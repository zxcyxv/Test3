"""
Router 진단 스크립트
1. Spatial Decision Boundary 히트맵
2. Expert Loss Attribution (L_A vs L_B)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 모델 로드
MODEL_PATH = Path('/workspace/SoccerPredict/open_track1/film_smoe_transformer.pt')

def main():
    print("=" * 70)
    print("Router 정밀 진단")
    print("=" * 70)

    # 1. 체크포인트 로드
    checkpoint = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)

    # 모델 재구성 (current pipeline)
    from main_train import (
        FiLMSpatialMoETransformer, prepare_sequence_data,
        create_boundary_labels, FIELD_X, FIELD_Y
    )
    import pandas as pd

    # 데이터 로드
    DATA_DIR = Path('/workspace/SoccerPredict/open_track1')
    df = pd.read_csv(DATA_DIR / 'train_features_v2.csv')
    (coords, cont, angles, cat, valid_mask,
     cont_valid_mask, cont_idx, angle_idx, cat_idx) = prepare_sequence_data(df)

    # 타겟 및 경계 정보 추출
    y = df[['target_end_x', 'target_end_y']].values
    boundary_zone = create_boundary_labels(y[:, 0], y[:, 1])

    # Train/Val split (동일한 시드 사용)
    from sklearn.model_selection import train_test_split
    (coords_train, coords_val,
     cont_train, cont_val,
     angles_train, angles_val,
     cat_train, cat_val,
     valid_mask_train, valid_mask_val,
     cont_valid_mask_train, cont_valid_mask_val,
     y_train, y_val,
     zone_train, zone_val) = train_test_split(
        coords, cont, angles, cat, valid_mask, cont_valid_mask, y, boundary_zone,
        test_size=0.2, random_state=42
    )

    # 스케일링 적용 (checkpoint에서 cont_mean/std 로드)
    if 'cont_mean' in checkpoint and 'cont_std' in checkpoint:
        cont_mean = checkpoint['cont_mean']
        cont_std = checkpoint['cont_std']
    else:
        cont_mean = checkpoint['scaler_mean']
        cont_std = checkpoint['scaler_scale']

    coord_scale = checkpoint.get('coord_scale', np.array([FIELD_X, FIELD_Y]))

    cont_val = (cont_val - cont_mean) / cont_std
    cont_val[~cont_valid_mask_val] = 0.0
    cont_val = np.concatenate([cont_val, angles_val], axis=-1)

    padding_mask_val = ~valid_mask_val

    # 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FiLMSpatialMoETransformer(
        num_cont_features=cont_val.shape[2],
        cat_cardinalities=tuple(int(cat_train[:, :, i].max()) + 1 for i in range(cat_train.shape[2])),
        d_model=128,
        nhead=4,
        num_layers=2,
        dropout=0.2,
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    coords_val_t = torch.FloatTensor(coords_val).to(device)
    cont_val_t = torch.FloatTensor(cont_val).to(device)
    cat_val_t = torch.LongTensor(cat_val).to(device)
    padding_mask_val_t = torch.BoolTensor(padding_mask_val).to(device)
    y_val_t = torch.FloatTensor(y_val).to(device)

    print(f"\nValidation samples: {len(coords_val)}")
    print(f"Zone distribution: {np.bincount(zone_val)}")

    # =========================================================================
    # 진단 1: Spatial Decision Boundary 히트맵
    # =========================================================================
    print("\n" + "=" * 70)
    print("[1] Spatial Decision Boundary 히트맵")
    print("=" * 70)

    with torch.no_grad():
        outputs = model(coords_val_t, cont_val_t, cat_val_t, padding_mask_val_t, temperature=0.05, return_all=True)

        gate = outputs['gate'].cpu().numpy().flatten()
        mu_A = outputs['mu_A'].cpu().numpy()
        mu_B = outputs['mu_B'].cpu().numpy()
        mu = outputs['mu'].cpu().numpy()
        pi_A = outputs['pi_A'].cpu().numpy()
        pi_B = outputs['pi_B'].cpu().numpy()
        pi = outputs['pi'].cpu().numpy()
        y_np = y_val

        # start_x_7, start_y_7 추출
        start_x = coords_val[:, -1, 0] * coord_scale[0]
        start_y = coords_val[:, -1, 1] * coord_scale[1]

        # end_x, end_y (target)
        end_x = y_val[:, 0]
        end_y = y_val[:, 1]

    # 히트맵 1: START 위치 기준 Gate 분포
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1-1: Gate by START position
    ax = axes[0, 0]
    scatter = ax.scatter(start_x, start_y, c=gate, cmap='RdYlBu_r',
                         alpha=0.5, s=5, vmin=0, vmax=1)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_xlabel('start_x')
    ax.set_ylabel('start_y')
    ax.set_title('Gate by START position')
    ax.axvline(5, color='red', linestyle='--', alpha=0.5)
    ax.axvline(100, color='red', linestyle='--', alpha=0.5)
    ax.axhline(5, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(63, color='blue', linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=ax, label='Gate (0=InField, 1=Boundary)')

    # 1-2: Gate by END position (target)
    ax = axes[0, 1]
    scatter = ax.scatter(end_x, end_y, c=gate, cmap='RdYlBu_r',
                         alpha=0.5, s=5, vmin=0, vmax=1)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 68)
    ax.set_xlabel('end_x (target)')
    ax.set_ylabel('end_y (target)')
    ax.set_title('Gate by END position (TARGET)')
    ax.axvline(5, color='red', linestyle='--', alpha=0.5)
    ax.axvline(100, color='red', linestyle='--', alpha=0.5)
    ax.axhline(5, color='blue', linestyle='--', alpha=0.5)
    ax.axhline(63, color='blue', linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=ax, label='Gate (0=InField, 1=Boundary)')

    # 1-3: Gate by Zone (box plot)
    ax = axes[0, 2]
    zone_names = ['In-field', 'Top-out', 'Bottom-out', 'Goal-line']
    gate_by_zone = [gate[zone_val == i] for i in range(4)]
    bp = ax.boxplot(gate_by_zone, labels=zone_names)
    ax.set_ylabel('Gate Value')
    ax.set_title('Gate Distribution by Zone')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)

    # Zone별 통계 출력
    print("\n[Gate by Zone - 상세 통계]")
    for i, name in enumerate(zone_names):
        g = gate[zone_val == i]
        print(f"  {name:12s}: mean={g.mean():.4f}, std={g.std():.4f}, "
              f"min={g.min():.4f}, max={g.max():.4f}, "
              f"<0.3: {(g < 0.3).mean()*100:.1f}%, >0.7: {(g > 0.7).mean()*100:.1f}%")

    # =========================================================================
    # 진단 2: Expert Loss Attribution
    # =========================================================================
    print("\n" + "=" * 70)
    print("[2] Expert Loss Attribution (L_A vs L_B)")
    print("=" * 70)

    # L_A = ||y - μ_A||², L_B = ||y - μ_B||² (use best mode)
    batch_idx = np.arange(len(y_np))
    best_A = pi_A.argmax(axis=1)
    best_B = pi_B.argmax(axis=1)
    best_M = pi.argmax(axis=1)
    mu_A_best = mu_A[batch_idx, best_A] * coord_scale
    mu_B_best = mu_B[batch_idx, best_B] * coord_scale
    mu_best = mu[batch_idx, best_M] * coord_scale
    L_A = np.sum((y_np - mu_A_best) ** 2, axis=1)
    L_B = np.sum((y_np - mu_B_best) ** 2, axis=1)
    L_mixed = np.sum((y_np - mu_best) ** 2, axis=1)

    # 2-1: L_A vs L_B scatter (by zone)
    ax = axes[1, 0]
    colors = ['green', 'orange', 'purple', 'red']
    for i, (name, color) in enumerate(zip(zone_names, colors)):
        mask = zone_val == i
        ax.scatter(L_A[mask], L_B[mask], c=color, alpha=0.3, s=5, label=name)
    ax.plot([0, 2000], [0, 2000], 'k--', alpha=0.5, label='L_A = L_B')
    ax.set_xlabel('L_A (In-field Expert Loss)')
    ax.set_ylabel('L_B (Boundary Expert Loss)')
    ax.set_title('Expert Loss Attribution')
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)
    ax.legend()

    # 2-2: Expert preference ratio by zone
    ax = axes[1, 1]
    # L_B < L_A means Boundary expert is better
    better_B_ratio = [(L_B[zone_val == i] < L_A[zone_val == i]).mean() * 100
                      for i in range(4)]
    bars = ax.bar(zone_names, better_B_ratio, color=colors)
    ax.set_ylabel('% samples where L_B < L_A')
    ax.set_title('Expert B (Boundary) Preference by Zone')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    for bar, ratio in zip(bars, better_B_ratio):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{ratio:.1f}%', ha='center', va='bottom')

    # Zone별 상세 통계
    print("\n[Expert Loss by Zone]")
    for i, name in enumerate(zone_names):
        mask = zone_val == i
        la = L_A[mask]
        lb = L_B[mask]
        lm = L_mixed[mask]
        better_b = (lb < la).mean() * 100
        print(f"  {name:12s}: L_A={la.mean():.1f}±{la.std():.1f}, "
              f"L_B={lb.mean():.1f}±{lb.std():.1f}, "
              f"L_mix={lm.mean():.1f}±{lm.std():.1f}, "
              f"B_better={better_b:.1f}%")

    # 2-3: Gate vs Loss Difference
    ax = axes[1, 2]
    loss_diff = L_A - L_B  # positive = B is better
    scatter = ax.scatter(loss_diff, gate, c=zone_val, cmap='tab10',
                         alpha=0.3, s=5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('L_A - L_B (positive = B better)')
    ax.set_ylabel('Gate Value')
    ax.set_title('Gate vs Expert Preference')

    # 상관관계 계산
    from scipy import stats
    corr, pval = stats.pearsonr(loss_diff, gate)
    print(f"\n[Gate vs (L_A - L_B) 상관관계]")
    print(f"  Pearson r = {corr:.4f}, p-value = {pval:.2e}")
    print(f"  해석: {'양의 상관 (올바른 방향)' if corr > 0 else '음의 상관 또는 무관'}")

    plt.tight_layout()
    plt.savefig('/workspace/SoccerPredict/router_diagnosis.png', dpi=150)
    print(f"\n히트맵 저장: /workspace/SoccerPredict/router_diagnosis.png")

    # =========================================================================
    # 진단 3: Grid-based Gate 히트맵 (합성 데이터)
    # =========================================================================
    print("\n" + "=" * 70)
    print("[3] Synthetic Grid Gate 히트맵")
    print("=" * 70)

    # 경기장 전체를 그리드로 나누어 Gate 값 계산
    # 단, 이를 위해선 전체 시퀀스가 필요하므로, 가상의 시퀀스 생성
    # 마지막 이벤트의 start_x, start_y만 변경하고 나머지는 평균값 사용

    grid_size = 21  # 21x21 그리드
    x_grid = np.linspace(0, 105, grid_size)
    y_grid = np.linspace(0, 68, grid_size)

    # 기준 시퀀스: 평균적인 시퀀스 사용 (이미 스케일링됨)
    base_coords = coords_val[0].copy()
    base_cont = cont_val[0].copy()
    base_cat = cat_val[0].copy()
    base_mask = padding_mask_val[0].copy()

    gate_grid = np.zeros((grid_size, grid_size))

    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            test_coords = base_coords.copy()
            test_coords[-1, 0] = x / coord_scale[0]
            test_coords[-1, 1] = y / coord_scale[1]

            test_t_coords = torch.FloatTensor(test_coords).unsqueeze(0).to(device)
            test_t_cont = torch.FloatTensor(base_cont).unsqueeze(0).to(device)
            test_t_cat = torch.LongTensor(base_cat).unsqueeze(0).to(device)
            test_t_mask = torch.BoolTensor(base_mask).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(test_t_coords, test_t_cont, test_t_cat, test_t_mask, temperature=0.05, return_all=True)
                gate_grid[j, i] = out['gate'].item()

    # 히트맵 그리기
    fig2, ax = plt.subplots(1, 1, figsize=(12, 8))
    im = ax.imshow(gate_grid, extent=[0, 105, 0, 68], origin='lower',
                   cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    ax.set_xlabel('start_x')
    ax.set_ylabel('start_y')
    ax.set_title('Synthetic Grid: Gate(x, y) - Expected: Bright at boundaries')

    # 경계선 표시
    ax.axvline(5, color='white', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(100, color='white', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(5, color='white', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(63, color='white', linestyle='--', linewidth=2, alpha=0.7)

    plt.colorbar(im, ax=ax, label='Gate (0=InField, 1=Boundary)')
    plt.tight_layout()
    plt.savefig('/workspace/SoccerPredict/router_grid_heatmap.png', dpi=150)
    print(f"Grid 히트맵 저장: /workspace/SoccerPredict/router_grid_heatmap.png")

    # Grid 통계
    print(f"\n[Grid Gate 통계]")
    print(f"  전체: mean={gate_grid.mean():.4f}, std={gate_grid.std():.4f}")
    print(f"  중앙 (x:20-85, y:10-58): mean={gate_grid[3:17, 4:17].mean():.4f}")
    print(f"  좌측 경계 (x<5): mean={gate_grid[:, 0].mean():.4f}")
    print(f"  우측 경계 (x>100): mean={gate_grid[:, -1].mean():.4f}")
    print(f"  상단 경계 (y>63): mean={gate_grid[-1, :].mean():.4f}")
    print(f"  하단 경계 (y<5): mean={gate_grid[0, :].mean():.4f}")

    # =========================================================================
    # 결론
    # =========================================================================
    print("\n" + "=" * 70)
    print("[결론]")
    print("=" * 70)

    # 진단 결과 요약
    gate_std = gate.std()
    gate_range = gate.max() - gate.min()
    expert_corr = corr

    issues = []
    if gate_std < 0.15:
        issues.append("- Gate std < 0.15: 라우터가 거의 상수값 출력 (Collapse)")
    if abs(expert_corr) < 0.1:
        issues.append("- |Corr(Gate, L_A-L_B)| < 0.1: 라우터가 전문가 성능과 무관하게 동작")
    if gate_grid.std() < 0.1:
        issues.append("- Grid Gate std < 0.1: 공간적 패턴 학습 실패")

    infield_gate = gate[zone_val == 0].mean()
    boundary_gate = gate[zone_val != 0].mean()
    if abs(infield_gate - boundary_gate) < 0.1:
        issues.append(f"- In-field({infield_gate:.3f}) ≈ Boundary({boundary_gate:.3f}): 영역 구분 실패")

    if issues:
        print("발견된 문제점:")
        for issue in issues:
            print(issue)
    else:
        print("라우터가 정상적으로 동작하는 것으로 보입니다.")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
