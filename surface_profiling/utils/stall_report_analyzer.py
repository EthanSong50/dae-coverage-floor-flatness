# surface_profiling/utils/stall_report_analyzer.py
"""
mission_execution/utils/stall_logger.py가 주행 중 nav2 feedback을 근거로 기록한
analytics/logs/stall_report_<mission_start_ts>.csv를, 같은 미션의 실제 주행 궤적
(analytics/paths/robot_path_<end_ts>.csv, 절대 타임스탬프 포함)과 좌표로 대조해서
"어디서, 얼마나, 왜" 지연됐는지 한눈에 보이는 시각화 + 짧은 텍스트 요약을 만듦.

mission_execution 패키지의 클래스를 import하지 않고 두 CSV의 문서화된 컬럼
포맷만 직접 파싱함 - 도입 배경은 HISTORY.md §5 참고.

호출 시점: surface_profiler.py의 run()이 히트맵(Stage 3)까지 만든 직후(Stage 4).
이 시점이면 mission_executor.py가 이미 미션을 끝내고 stall_report/robot_path
CSV를 다 써놓은 뒤이므로 파일이 존재함. 어떤 이유로든(파일 없음/포맷 어긋남
등) 실패해도 예외를 밖으로 던지지 않음 - 히트맵 파이프라인 자체를 막으면
안 되므로.
"""

import os
import re
import csv
import bisect

import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 비활성화 (헤드리스 환경 안전성 확보)
import matplotlib.pyplot as plt


_STALL_TS_RE = re.compile(r"stall_report_(\d+)\.csv$")
_PATH_TS_RE = re.compile(r"robot_path_(\d+)\.csv$")


def _find_latest_stall_report(log_dir):
    """stall_log_dir 안에서 파일명의 mission_start_ts가 가장 큰(=가장 최근 미션의)
    stall_report_*.csv를 찾음. mtime이 아니라 파일명 타임스탬프를 쓰는 이유는
    이 값이 곧 robot_path CSV와 대조할 mission_start 절대시각 그 자체이기 때문."""
    if not os.path.isdir(log_dir):
        return None, None
    best_ts, best_path = None, None
    for fname in os.listdir(log_dir):
        m = _STALL_TS_RE.match(fname)
        if m:
            ts = int(m.group(1))
            if best_ts is None or ts > best_ts:
                best_ts, best_path = ts, os.path.join(log_dir, fname)
    return best_ts, best_path


def _find_matching_robot_path(path_dir, mission_start_ts, max_delay_sec=180.0):
    """robot_path_*.csv 중 첫 데이터 행의 절대 timestamp가 mission_start_ts
    이후로 가장 가까운(=같은 미션에서 나온) 파일을 찾음. post_localization_wait_sec/
    start_prepass 등으로 몇 초 정도는 차이 날 수 있어 약간의 여유(-5s)와
    상한(max_delay_sec)을 둠."""
    if not os.path.isdir(path_dir):
        return None
    best_path, best_delta = None, None
    for fname in os.listdir(path_dir):
        if not _PATH_TS_RE.match(fname):
            continue
        full = os.path.join(path_dir, fname)
        try:
            with open(full, 'r') as f:
                next(f)  # header
                first_line = f.readline()
            first_ts = float(first_line.split(',')[0])
        except Exception:
            continue
        delta = first_ts - mission_start_ts
        if -5.0 <= delta <= max_delay_sec:
            if best_delta is None or delta < best_delta:
                best_path, best_delta = full, delta
    return best_path


def _load_path_history(robot_path_csv):
    rows = []
    with open(robot_path_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            rows.append((float(row[0]), float(row[1]), float(row[2])))
    rows.sort(key=lambda r: r[0])
    return rows


def _interpolate_position(path_history, times, t):
    idx = bisect.bisect_left(times, t)
    if idx <= 0:
        return path_history[0][1], path_history[0][2]
    if idx >= len(path_history):
        return path_history[-1][1], path_history[-1][2]
    t0, x0, y0 = path_history[idx - 1]
    t1, x1, y1 = path_history[idx]
    if t1 <= t0:
        return x0, y0
    frac = (t - t0) / (t1 - t0)
    return x0 + frac * (x1 - x0), y0 + frac * (y1 - y0)


def _load_stall_events(stall_csv):
    events = []
    with open(stall_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # "# stall threshold = ..." 주석 행
        next(reader)  # 헤더 행
        for row in reader:
            if not row:
                continue
            events.append({
                'label': row[0],
                'elapsed_sec': float(row[1]),
                'duration_sec': float(row[2]),
                'stalled_value': row[3],
                'recoveries': int(row[4]),
                'resolved': row[5].startswith('yes'),
            })
    return events


def analyze_and_visualize_stalls(workspace_root, mission_exec_cfg):
    """가장 최근 stall_report CSV를 분석해 analytics/logs에 텍스트 요약을,
    문제 구간이 하나라도 있으면 그 위치를 실제 주행 궤적 위에 표시한 PNG를
    visualization/mission_execution에 저장함. 실패해도 예외를 던지지 않음."""
    print("\n[Stage 4/4] Driving Stall Report Analysis")
    try:
        log_dir = os.path.join(workspace_root, mission_exec_cfg.get('stall_log_dir', 'analytics/logs'))
        path_dir = os.path.join(workspace_root, mission_exec_cfg.get('output_path_dir', 'analytics/paths'))
        vis_dir = os.path.join(workspace_root, mission_exec_cfg.get('visualization_dir', 'visualization/mission_execution'))

        mission_start_ts, stall_csv = _find_latest_stall_report(log_dir)
        if stall_csv is None:
            print(f"[!] No stall_report_*.csv found under {log_dir} — skipping stall analysis.")
            return
        report_name = os.path.basename(stall_csv)

        events = _load_stall_events(stall_csv)

        os.makedirs(log_dir, exist_ok=True)
        analysis_txt_path = os.path.join(log_dir, f"stall_analysis_{mission_start_ts}.txt")

        if not events:
            with open(analysis_txt_path, 'w') as f:
                f.write(f"# analysis of {report_name}\n0 stalls detected.\n")
            print(f"[+] No stalls (>= threshold) detected this mission — "
                  f"summary written to {analysis_txt_path}.")
            return

        total_dur = sum(e['duration_sec'] for e in events)
        unresolved = [e for e in events if not e['resolved']]
        with_recovery = [e for e in events if e['recoveries'] > 0]

        summary_lines = [
            f"# analysis of {report_name}",
            f"{len(events)} stall(s), total {total_dur:.1f}s stalled, "
            f"{len(unresolved)} unresolved (still stalled when action ended), "
            f"{len(with_recovery)} with nav2 recovery triggered during the stall",
            "",
        ]
        for i, e in enumerate(events, start=1):
            summary_lines.append(
                f"#{i} [{e['label']}] mission_t={e['elapsed_sec']:.1f}s "
                f"duration={e['duration_sec']:.1f}s recoveries={e['recoveries']} "
                f"resolved={'yes' if e['resolved'] else 'NO'}"
            )
        with open(analysis_txt_path, 'w') as f:
            f.write("\n".join(summary_lines) + "\n")
        print(f"[+] Stall analysis summary ({len(events)} events, {total_dur:.1f}s total) "
              f"written to {analysis_txt_path}.")

        robot_path_csv = _find_matching_robot_path(path_dir, mission_start_ts)
        if robot_path_csv is None:
            print(f"[!] No matching robot_path_*.csv found under {path_dir} for mission "
                  f"start {mission_start_ts} — skipping stall visualization (text summary still saved).")
            return

        path_history = _load_path_history(robot_path_csv)
        if len(path_history) < 2:
            print("[!] robot_path CSV too short — skipping stall visualization.")
            return

        times = [r[0] for r in path_history]
        for e in events:
            e['x'], e['y'] = _interpolate_position(path_history, times, mission_start_ts + e['elapsed_sec'])

        os.makedirs(vis_dir, exist_ok=True)
        img_out_path = os.path.join(vis_dir, f"robot_path_{mission_start_ts}_nav_stall.png")

        all_x = [r[1] for r in path_history]
        all_y = [r[2] for r in path_history]

        fig, ax = plt.subplots(figsize=(10, 9))
        ax.plot(all_x, all_y, 'gray', alpha=0.4, linewidth=1, label='Actual Driven Path')

        legend_seen = set()
        for i, e in enumerate(events, start=1):
            resolved = e['resolved']
            color = 'purple' if resolved else 'magenta'
            marker = 'o' if resolved else '*'
            size = 70 if resolved else 150
            legend_label = 'Nav2 Stall (resolved)' if resolved else 'Nav2 Stall (UNRESOLVED)'
            ax.scatter(e['x'], e['y'], c=color, marker=marker, s=size,
                       edgecolors='black', linewidths=0.8, zorder=6,
                       label=None if legend_label in legend_seen else legend_label)
            legend_seen.add(legend_label)
            recovery_mark = '*R' if e['recoveries'] > 0 else ''
            ax.annotate(f"#{i} {e['duration_sec']:.1f}s{recovery_mark}", (e['x'], e['y']),
                        fontsize=7, color='black', xytext=(4, 4), textcoords='offset points')

        ax.set_title(f'Nav2 Feedback-Based Stall Points ({len(events)} events, '
                     f'total {total_dur:.1f}s, {len(unresolved)} unresolved)')
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True); ax.axis('equal')

        # 분석이 짧으므로 별도 파일 대신 그림 아래에 번호별 상세를 그대로 붙임.
        detail_text = "\n".join(
            f"#{i} [{e['label']}] t={e['elapsed_sec']:.1f}s dur={e['duration_sec']:.1f}s "
            f"recoveries={e['recoveries']} {'resolved' if e['resolved'] else 'UNRESOLVED'}"
            for i, e in enumerate(events, start=1)
        )
        n_lines = detail_text.count("\n") + 2
        # 0.13 =  x축 눈금/라벨("X (m)")이 축 아래에 이미 차지하는 여유분.
        # 그 아래(figure 맨 밑, y=0.02)에 상세 목록을 고정 배치해 겹치지 않게 함.
        bottom_margin = min(0.45, 0.13 + 0.022 * n_lines)
        fig.subplots_adjust(bottom=bottom_margin)
        fig.text(0.02, 0.02, detail_text, fontsize=6.5,
                  family='monospace', va='bottom', ha='left')

        fig.savefig(img_out_path, dpi=300)
        plt.close(fig)
        print(f"[+] Stall visualization saved to {img_out_path} "
              f"({len(events)} events plotted, {len(unresolved)} unresolved, "
              f"{len(with_recovery)} with nav2 recovery).")
    except Exception as exc:
        print(f"[-] Stall analysis failed (non-fatal, heatmap pipeline unaffected): {exc}")
