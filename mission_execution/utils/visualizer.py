# mission_execution/utils/visualizer.py

import csv
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 비활성화 (헤드리스 환경/Jetson 안전성 확보)
import matplotlib.pyplot as plt

import os
import math
import numpy as np
import cv2
import yaml as _yaml

def _load_occupancy_grid(map_yaml_path):
    with open(map_yaml_path, 'r') as f:
        map_data = _yaml.safe_load(f)
    resolution = map_data['resolution']
    origin = map_data['origin']
    image_path = os.path.join(os.path.dirname(map_yaml_path), map_data['image'])
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Map image not found: {image_path}")
    return img, resolution, origin

def _meter_to_pixel(x, y, origin, resolution, map_height):
    px = int(round((x - origin[0]) / resolution))
    py = int(round(map_height - 1 - (y - origin[1]) / resolution))
    return px, py

def visualize_planned_wall_proximity(json_path_data, map_yaml_path, img_out_path,
                                      robot_radius_m=0.15, sample_step_m=0.05):
    """
    planning된 path가 벽으로부터 robot_radius_m 이내로 지나가야 하는 지점을 찾아
    별도 PNG로 저장함.
    연속된 두 웨이포인트 사이를 sample_step_m 간격으로 보간해서 검사함.
    F2C 스와스 웨이포인트는 앵커(시작/꼭짓점/끝)만 남기고 직선 중간의
    보간점을 두지 않으므로(HISTORY.md §2 참고), 웨이포인트만 봐서는 직선
    중간의 위험 지점을 놓칠 수 있기 때문임.
    """
    print("\n[*] Generating Planned-Path Wall-Proximity Risk Map...")
    try:
        img, resolution, origin = _load_occupancy_grid(map_yaml_path)
        free_mask = np.where(img >= 200, 255, 0).astype(np.uint8)
        dist_px = cv2.distanceTransform(free_mask, cv2.DIST_L2, 5)
        map_height = img.shape[0]

        planned_x = [wp['pose']['position']['x'] for wp in json_path_data]
        planned_y = [wp['pose']['position']['y'] for wp in json_path_data]

        risk_x, risk_y = [], []
        for i in range(len(planned_x) - 1):
            x1, y1, x2, y2 = planned_x[i], planned_y[i], planned_x[i + 1], planned_y[i + 1]
            seg_len = math.hypot(x2 - x1, y2 - y1)
            n_samples = max(1, int(seg_len / sample_step_m))
            for k in range(n_samples + 1):
                t = k / n_samples if n_samples > 0 else 0.0
                sx, sy = x1 + t * (x2 - x1), y1 + t * (y2 - y1)
                px, py = _meter_to_pixel(sx, sy, origin, resolution, map_height)
                if 0 <= px < dist_px.shape[1] and 0 <= py < dist_px.shape[0]:
                    if dist_px[py, px] * resolution < robot_radius_m:
                        risk_x.append(sx)
                        risk_y.append(sy)

        plt.figure(figsize=(10, 8))
        plt.plot(planned_x, planned_y, 'b--', label='Planned Path', alpha=0.4, linewidth=1)
        if risk_x:
            plt.scatter(risk_x, risk_y, c='orange', marker='x', s=40,
                        label=f'Wall clearance < {robot_radius_m * 100:.0f}cm', zorder=5)
        plt.title(f'Planned Path — Points Needing < {robot_radius_m * 100:.0f}cm Wall Clearance')
        plt.xlabel('X (m)'); plt.ylabel('Y (m)')
        plt.legend(); plt.grid(True); plt.axis('equal')
        plt.savefig(img_out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Risk map saved to '{img_out_path}' ({len(risk_x)} risky sample points).")
    except Exception as e:
        print(f"[-] Failed to generate risk map: {e}")

def visualize_stall_points(csv_filename, img_out_path,
                            stall_speed_mps=0.03, stall_min_duration_sec=4.0):
    """
    실제 AMCL 궤적(csv)에서 속도가 stall_speed_mps 미만으로
    stall_min_duration_sec 이상 지속된 구간을 '멈춤/지연'으로 표시함.

    다만, 이 CSV엔 정상적인 단일점(_single) 정차 캡처처럼 '의도된
    정지'도 섞여 있음. 이 함수는 순수 속도 기반이라 의도된 정지와 코너에서
    막혀 생긴 비정상 정지를 자동으로 구분하지 못함 - 결과를 final_path의
    coverage 단일점 좌표와 눈으로 대조해서 한 번 더 걸러야 함. nav2
    feedback 기반으로 "왜"까지 남기는 상호 보완적 매커니즘은
    utils/stall_logger.py와 surface_profiling/utils/stall_report_analyzer.py
    참고(도입 배경은 HISTORY.md §4/§5).
    """
    print("\n[*] Generating Actual-Path Stall/Delay Map...")
    try:
        rows = []
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                rows.append((float(row[0]), float(row[1]), float(row[2])))

        # _amcl_monitor_callback의 이중 기록 버그로 생기는 근접 중복 타임스탬프 제거
        rows.sort(key=lambda r: r[0])
        dedup = []
        for t, x, y in rows:
            if dedup and t - dedup[-1][0] < 1e-3:
                continue
            dedup.append((t, x, y))
        rows = dedup

        if len(rows) < 2:
            print("[!] Not enough path_history samples to analyze stalls.")
            return

        all_x, all_y = [r[1] for r in rows], [r[2] for r in rows]
        stall_segments = []
        cur_run = [rows[0]]

        for i in range(1, len(rows)):
            t0, x0, y0 = rows[i - 1]
            t1, x1, y1 = rows[i]
            dt = t1 - t0
            if dt <= 0:
                continue
            speed = math.hypot(x1 - x0, y1 - y0) / dt
            if speed < stall_speed_mps:
                cur_run.append(rows[i])
            else:
                if len(cur_run) >= 2 and (cur_run[-1][0] - cur_run[0][0]) >= stall_min_duration_sec:
                    stall_segments.append(cur_run)
                cur_run = [rows[i]]
        if len(cur_run) >= 2 and (cur_run[-1][0] - cur_run[0][0]) >= stall_min_duration_sec:
            stall_segments.append(cur_run)

        plt.figure(figsize=(10, 8))
        plt.plot(all_x, all_y, 'gray', alpha=0.4, linewidth=1, label='Actual Driven Path')

        total_stall_time = 0.0
        for idx, seg in enumerate(stall_segments):
            sx, sy = [p[1] for p in seg], [p[2] for p in seg]
            duration = seg[-1][0] - seg[0][0]
            total_stall_time += duration
            plt.scatter(sx, sy, c='red', s=30, zorder=5,
                        label=f'Stall/Delay (>{stall_min_duration_sec:.1f}s)' if idx == 0 else None)
            plt.annotate(f"{duration:.1f}s", (sx[0], sy[0]), fontsize=7, color='darkred')

        plt.title(f'Actual Path — Stall/Delay Points ({len(stall_segments)} events, '
                  f'total {total_stall_time:.1f}s)')
        plt.xlabel('X (m)'); plt.ylabel('Y (m)')
        plt.legend(); plt.grid(True); plt.axis('equal')
        plt.savefig(img_out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Stall map saved to '{img_out_path}' "
              f"({len(stall_segments)} events, {total_stall_time:.1f}s total).")
    except Exception as e:
        print(f"[-] Failed to generate stall map: {e}")

def visualize_paths(csv_filename, json_path_data, img_out_path):
    """
    실제 AMCL 주행 데이터(CSV)와 계획된 경로(JSON)를 비교 시각화해 PNG로 저장함.
    """
    print("\n[*] Generating Path Tracking Performance Graph...")
    try:
        # 1. JSON 파싱 (계획된 웨이포인트)
        planned_x = [wp['pose']['position']['x'] for wp in json_path_data]
        planned_y = [wp['pose']['position']['y'] for wp in json_path_data]

        # 2. CSV 파싱 (실제 AMCL 주행 궤적)
        actual_x = []
        actual_y = []
        with open(csv_filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                actual_x.append(float(row[1]))
                actual_y.append(float(row[2]))

        # 3. 그래프 그리기 (맵 원점 기준 1:1 매칭)
        plt.figure(figsize=(10, 8))

        # 계획된 경로 (파란색 점선)
        plt.plot(planned_x, planned_y, 'b--o', label='Planned Path (Waypoints)', markersize=4, alpha=0.6)

        # 실제 주행 궤적 (빨간색 실선)
        plt.plot(actual_x, actual_y, 'r-', label='Actual Driven Path (AMCL)', linewidth=2)

        # 시작점과 목표점 강조
        plt.plot(planned_x[0], planned_y[0], 'go', label='Start', markersize=8)
        plt.plot(planned_x[-1], planned_y[-1], 'ko', label='Goal', markersize=8)

        plt.title('Nav2 Path Tracking Performance (Map Frame)')
        plt.xlabel('X coordinate (m)')
        plt.ylabel('Y coordinate (m)')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')  # 맵 비율 유지 (왜곡 방지)

        # 4. 이미지 저장 (GUI 에러 방지)
        plt.savefig(img_out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Visualization successfully saved to '{img_out_path}'.")
    except Exception as e:
        print(f"[-] Failed to generate visualization: {e}")