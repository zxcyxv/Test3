"""
데이터 분포 분석: 필드 위치별 패스 패턴
X: 0 (수비) ~ 105 (공격)
Y: 0 (사이드) ~ 68 (사이드), 34 = 중앙
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def log(msg):
    print(msg, flush=True)

DATA_DIR = Path('/workspace/SoccerPredict/open_track1')
FIELD_X = 105
FIELD_Y = 68


def main():
    log("Loading data...")
    df = pd.read_csv(DATA_DIR / 'train.csv')

    # 각 에피소드의 마지막 이벤트만 추출 (항상 Pass)
    last_events = df.groupby('game_episode').tail(1).copy()
    log(f"Total episodes: {len(last_events)}")

    # Success / Fail 분리
    success = last_events[last_events['result_name'] == 'Successful']
    fail = last_events[last_events['result_name'] != 'Successful']

    log(f"Success: {len(success)} ({len(success)/len(last_events)*100:.1f}%)")
    log(f"Fail: {len(fail)} ({len(fail)/len(last_events)*100:.1f}%)")

    # ============================================
    # 1. 기본 통계
    # ============================================
    log("\n" + "=" * 60)
    log("[1] 마지막 패스 좌표 기본 통계")
    log("=" * 60)

    for name, data in [("All", last_events), ("Success", success), ("Fail", fail)]:
        log(f"\n{name} (n={len(data)}):")
        log(f"  start_x: mean={data['start_x'].mean():.1f}, std={data['start_x'].std():.1f}")
        log(f"  start_y: mean={data['start_y'].mean():.1f}, std={data['start_y'].std():.1f}")
        log(f"  end_x:   mean={data['end_x'].mean():.1f}, std={data['end_x'].std():.1f}")
        log(f"  end_y:   mean={data['end_y'].mean():.1f}, std={data['end_y'].std():.1f}")

    # ============================================
    # 2. 경계 영역 분석
    # ============================================
    log("\n" + "=" * 60)
    log("[2] 경계 영역 분석 (end_x, end_y 기준)")
    log("=" * 60)

    # X 경계 (0~10: 수비 깊숙이, 95~105: 공격 깊숙이)
    log("\n[X축 경계 - end_x 기준]")
    for name, data in [("Success", success), ("Fail", fail)]:
        defense_deep = (data['end_x'] < 10).mean() * 100
        attack_deep = (data['end_x'] > 95).mean() * 100
        mid = ((data['end_x'] >= 10) & (data['end_x'] <= 95)).mean() * 100
        log(f"  {name}: 수비깊숙이(<10)={defense_deep:.1f}%, 공격깊숙이(>95)={attack_deep:.1f}%, 중간={mid:.1f}%")

    # Y 경계 (0~10, 58~68: 사이드, 나머지: 중앙)
    log("\n[Y축 경계 - end_y 기준]")
    for name, data in [("Success", success), ("Fail", fail)]:
        left_side = (data['end_y'] < 10).mean() * 100
        right_side = (data['end_y'] > 58).mean() * 100
        center = ((data['end_y'] >= 10) & (data['end_y'] <= 58)).mean() * 100
        log(f"  {name}: 왼쪽사이드(<10)={left_side:.1f}%, 오른쪽사이드(>58)={right_side:.1f}%, 중앙={center:.1f}%")

    # ============================================
    # 3. 코너/골라인 근처 분석
    # ============================================
    log("\n" + "=" * 60)
    log("[3] 특수 영역 분석")
    log("=" * 60)

    # 상대 골대 근처 (x > 90, 25 < y < 43: 페널티 에어리어)
    log("\n[상대 페널티 에어리어 근처 (end_x > 90, 25 < end_y < 43)]")
    for name, data in [("Success", success), ("Fail", fail)]:
        in_box = ((data['end_x'] > 90) & (data['end_y'] > 25) & (data['end_y'] < 43)).mean() * 100
        log(f"  {name}: {in_box:.1f}%")

    # 코너 영역 (x < 10 or x > 95) and (y < 10 or y > 58)
    log("\n[코너 영역 (x 경계 AND y 경계)]")
    for name, data in [("Success", success), ("Fail", fail)]:
        corner = (((data['end_x'] < 10) | (data['end_x'] > 95)) &
                  ((data['end_y'] < 10) | (data['end_y'] > 58))).mean() * 100
        log(f"  {name}: {corner:.1f}%")

    # ============================================
    # 4. 패스 방향 분석
    # ============================================
    log("\n" + "=" * 60)
    log("[4] 패스 방향 분석")
    log("=" * 60)

    for name, data in [("Success", success), ("Fail", fail)]:
        dx = data['end_x'] - data['start_x']
        dy = data['end_y'] - data['start_y']

        forward = (dx > 5).mean() * 100  # 전진 패스
        backward = (dx < -5).mean() * 100  # 후진 패스
        lateral = ((dx >= -5) & (dx <= 5)).mean() * 100  # 횡패스

        log(f"\n{name}:")
        log(f"  전진(dx>5): {forward:.1f}%, 후진(dx<-5): {backward:.1f}%, 횡패스: {lateral:.1f}%")
        log(f"  평균 dx: {dx.mean():.1f}m, 평균 dy: {dy.mean():.1f}m")
        log(f"  평균 거리: {np.sqrt(dx**2 + dy**2).mean():.1f}m")

    # ============================================
    # 5. start_x 구간별 end_x 분포
    # ============================================
    log("\n" + "=" * 60)
    log("[5] 시작 위치별 종료 위치 패턴")
    log("=" * 60)

    zones = [(0, 35, "수비진영"), (35, 70, "중앙"), (70, 105, "공격진영")]

    for zone_start, zone_end, zone_name in zones:
        log(f"\n시작위치: {zone_name} (start_x: {zone_start}~{zone_end})")

        for name, data in [("Success", success), ("Fail", fail)]:
            zone_data = data[(data['start_x'] >= zone_start) & (data['start_x'] < zone_end)]
            if len(zone_data) > 0:
                log(f"  {name} (n={len(zone_data)}): end_x 평균={zone_data['end_x'].mean():.1f}, end_y 평균={zone_data['end_y'].mean():.1f}")

    # ============================================
    # 6. 시각화
    # ============================================
    log("\n시각화 생성 중...")

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # 1. end_x, end_y 히스토그램
    axes[0, 0].hist(success['end_x'], bins=50, alpha=0.7, label='Success', density=True)
    axes[0, 0].hist(fail['end_x'], bins=50, alpha=0.7, label='Fail', density=True)
    axes[0, 0].set_xlabel('end_x (m)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('end_x Distribution')
    axes[0, 0].legend()
    axes[0, 0].axvline(52.5, color='gray', linestyle='--', alpha=0.5, label='Midfield')

    axes[0, 1].hist(success['end_y'], bins=50, alpha=0.7, label='Success', density=True)
    axes[0, 1].hist(fail['end_y'], bins=50, alpha=0.7, label='Fail', density=True)
    axes[0, 1].set_xlabel('end_y (m)')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('end_y Distribution')
    axes[0, 1].legend()
    axes[0, 1].axvline(34, color='gray', linestyle='--', alpha=0.5)

    # 2. 2D 히트맵 (Success)
    h1 = axes[1, 0].hist2d(success['end_x'], success['end_y'], bins=30, cmap='Blues')
    axes[1, 0].set_xlabel('end_x (m)')
    axes[1, 0].set_ylabel('end_y (m)')
    axes[1, 0].set_title('Success Pass End Position Heatmap')
    plt.colorbar(h1[3], ax=axes[1, 0])
    # 골대 표시
    axes[1, 0].axvline(0, color='red', linewidth=2)
    axes[1, 0].axvline(105, color='red', linewidth=2)

    # 3. 2D 히트맵 (Fail)
    h2 = axes[1, 1].hist2d(fail['end_x'], fail['end_y'], bins=30, cmap='Oranges')
    axes[1, 1].set_xlabel('end_x (m)')
    axes[1, 1].set_ylabel('end_y (m)')
    axes[1, 1].set_title('Fail Pass End Position Heatmap')
    plt.colorbar(h2[3], ax=axes[1, 1])
    axes[1, 1].axvline(0, color='red', linewidth=2)
    axes[1, 1].axvline(105, color='red', linewidth=2)

    # 4. start -> end 화살표 (샘플)
    sample_s = success.sample(min(200, len(success)), random_state=42)
    sample_f = fail.sample(min(200, len(fail)), random_state=42)

    for _, row in sample_s.iterrows():
        axes[2, 0].arrow(row['start_x'], row['start_y'],
                         row['end_x'] - row['start_x'], row['end_y'] - row['start_y'],
                         head_width=1, head_length=1, fc='blue', ec='blue', alpha=0.3)
    axes[2, 0].set_xlim(0, 105)
    axes[2, 0].set_ylim(0, 68)
    axes[2, 0].set_xlabel('X (m)')
    axes[2, 0].set_ylabel('Y (m)')
    axes[2, 0].set_title('Success Pass Directions (sample)')
    axes[2, 0].set_aspect('equal')

    for _, row in sample_f.iterrows():
        axes[2, 1].arrow(row['start_x'], row['start_y'],
                         row['end_x'] - row['start_x'], row['end_y'] - row['start_y'],
                         head_width=1, head_length=1, fc='orange', ec='orange', alpha=0.3)
    axes[2, 1].set_xlim(0, 105)
    axes[2, 1].set_ylim(0, 68)
    axes[2, 1].set_xlabel('X (m)')
    axes[2, 1].set_ylabel('Y (m)')
    axes[2, 1].set_title('Fail Pass Directions (sample)')
    axes[2, 1].set_aspect('equal')

    plt.tight_layout()
    plt.savefig(DATA_DIR / 'data_distribution.png', dpi=150)
    log(f"저장 완료: {DATA_DIR / 'data_distribution.png'}")

    # ============================================
    # 7. 경계값 정확한 분포
    # ============================================
    log("\n" + "=" * 60)
    log("[7] 경계값 상세 분포")
    log("=" * 60)

    # end_x 극단값
    log("\n[end_x 극단값]")
    log(f"  end_x == 0: {(last_events['end_x'] == 0).sum()}개")
    log(f"  end_x < 1: {(last_events['end_x'] < 1).sum()}개")
    log(f"  end_x > 104: {(last_events['end_x'] > 104).sum()}개")
    log(f"  end_x == 105: {(last_events['end_x'] == 105).sum()}개")

    # end_y 극단값
    log("\n[end_y 극단값]")
    log(f"  end_y == 0: {(last_events['end_y'] == 0).sum()}개")
    log(f"  end_y < 1: {(last_events['end_y'] < 1).sum()}개")
    log(f"  end_y > 67: {(last_events['end_y'] > 67).sum()}개")
    log(f"  end_y == 68: {(last_events['end_y'] == 68).sum()}개")

    # 경계 근처에서 Success vs Fail 비율
    log("\n[경계 근처 Success 비율]")
    edge_x_low = last_events[last_events['end_x'] < 5]
    edge_x_high = last_events[last_events['end_x'] > 100]
    edge_y_low = last_events[last_events['end_y'] < 5]
    edge_y_high = last_events[last_events['end_y'] > 63]

    for name, data in [("end_x < 5", edge_x_low), ("end_x > 100", edge_x_high),
                       ("end_y < 5", edge_y_low), ("end_y > 63", edge_y_high)]:
        if len(data) > 0:
            success_rate = (data['result_name'] == 'Successful').mean() * 100
            log(f"  {name}: {len(data)}개, Success율={success_rate:.1f}%")


if __name__ == '__main__':
    main()
