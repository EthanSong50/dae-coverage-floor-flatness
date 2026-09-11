# mission_generation/mission_planning/mission_planner.py

import json
import numpy as np
import cv2
import os
import math
import time

from mission_planning.algorithms import tsp, coverage, transit
from mission_planning.utils import visualizer, geometry, sampler
from mission_planning import translator

class MissionPlanner:
    # 파라미터 업데이트
    def __init__(self, topomap_path, visualization_dir="./debug", robot_width=0.28, path_safety_margin=0.25, lidar_range=8.4, overlap=0.2, turn_weight=2.0, wall_weight=5.0, lidar_mount_height=0.338, lidar_vertical_fov_deg=15.0,
             blind_radius_m=None, boundary_repass_distance_m=1.5, enable_boundary_repass=True,
             enable_pendant_reorder=True, enable_entry_hint_ordering=True, enable_path_simplification=True, **kwargs):
        if not os.path.exists(topomap_path):
            raise FileNotFoundError(f"[!] Topomap file not found at: {topomap_path}")

        data = np.load(topomap_path, allow_pickle=True)

        masks = data['nodes']
        nondriveable_masks = data.get('nondriveable_nodes', [])

        self.map_resolution = float(data.get('resolution', 0.05))
        self.origin = data.get('origin', [0, 0])

        self.robot_width = robot_width
        self.path_safety_margin = path_safety_margin
        self.visualization_dir = os.path.abspath(visualization_dir)

        # mission_execution.boundary_repass_distance_m/enable_boundary_repass와
        # 동일한 값 - 실행 시 BoundaryRepassController가 실제로 로봇을 데려다
        # 놓을 위치(retrace 지점)를 계획 단계에서도 반영하기 위함
        # (_compute_repass_adjusted_exit 참고). 이 값은 시각화 미리보기뿐
        # 아니라 Step3의 current_pos(다음 노드로 가는 transit의 실제
        # 시작점)에도 반영되어 self.path_segments/final_path.json 자체를
        # 바꿈 - 계획-실행 값 불일치 시 위험, 도입 경위는 HISTORY.md §2 참고.
        self.boundary_repass_distance_m = boundary_repass_distance_m
        self.enable_boundary_repass = enable_boundary_repass

        # ablation 실험용 토글 3종 - 각 메커니즘의 기여도를 개별적으로 끄고
        # 측정하기 위함(paper.md 결론, 2026-09-03). 기본값은 모두 True(현재
        # 파이프라인 동작과 동일) - False로 두면 해당 메커니즘 없이 생성했을
        # final_path.json을 얻을 수 있음.
        self.enable_pendant_reorder = enable_pendant_reorder
        self.enable_entry_hint_ordering = enable_entry_hint_ordering
        self.enable_path_simplification = enable_path_simplification

        self.turn_weight = float(turn_weight)
        self.wall_weight = float(wall_weight)

        default_r = lidar_mount_height / math.tan(math.radians(lidar_vertical_fov_deg))
        self.blind_radius_m = blind_radius_m if blind_radius_m is not None else max(default_r, 1.0)
        self.blind_radius_px = self.blind_radius_m / self.map_resolution

        print(f"[*] map_resolution={self.map_resolution}, blind_radius_m={self.blind_radius_m:.3f}, blind_radius_px={self.blind_radius_px:.1f}")

        self.nodes = []
        for i in range(len(masks)):
            node_dict = {
                'id': i + 1,
                'driveable_mask': masks[i],
                'nondriveable_mask': nondriveable_masks[i] if i < len(nondriveable_masks) else None
            }
            self.nodes.append(node_dict)

        if not self.nodes:
            print("[ERROR] No nodes found in topomap file.")
            return

        # 전체 구동 가능 영역 병합 (A* 등에서 활용)
        self.global_mask = np.zeros_like(self.nodes[0]['driveable_mask'])
        for node in self.nodes:
            if node['driveable_mask'] is not None:
                self.global_mask = cv2.bitwise_or(self.global_mask, node['driveable_mask'])
        
        # 알고리즘 모듈들에 넘겨줄 로봇 파라미터 캡슐화
        self.robot_params = {
            'width_px': robot_width / self.map_resolution,
            'effective_swath_m': lidar_range * (1.0 - overlap),
            'swath_width_px': (lidar_range * (1.0 - overlap)) / self.map_resolution
        }
        
        # 결과 저장 리스트
        self.path_segments = []

        print(f"[*] MissionPlanner Initialized")

        if kwargs:
            print(f"[WARN] MissionPlanner received unexpected kwargs (ignored): {list(kwargs.keys())}")

    def generate_cost_map(self, mask):
        dist_px = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_m = dist_px * self.map_resolution
        
        cost_map = np.zeros_like(mask, dtype=np.uint8)
        
        # 1. 완벽한 안전 영역 (벽으로부터 로봇반경+마진 이상 떨어짐): 255
        safe_threshold = self.robot_width + self.path_safety_margin
        cost_map[dist_m >= safe_threshold] = 255
        
        # 2. 소프트 페널티 영역 (벽과 가깝지만 통과는 가능함, ex: 문지방): 50 ~ 250 그라데이션
        penalty_mask = (dist_m >= self.robot_width) & (dist_m < safe_threshold)
        if np.any(penalty_mask):
            normalized_dist = (dist_m[penalty_mask] - self.robot_width) / self.path_safety_margin
            cost_map[penalty_mask] = (50 + normalized_dist * 200).astype(np.uint8)
            
        # 3. 절대 불가 영역 (물리적 로봇 반경 이내): 0 (기본값이 0이므로 별도 대입 생략)
        return cost_map

    def generate_coverage_mask(self, mask):
        dist_px = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        dist_m = dist_px * self.map_resolution
        
        coverage_mask = np.zeros_like(mask, dtype=np.uint8)
        safe_threshold = self.robot_width + self.path_safety_margin
        
        coverage_mask[dist_m >= safe_threshold] = 255
        
        # 폴백 방어 로직: 맵이 너무 좁아서 마진 적용 시 영역이 아예 사라지면 로봇 반경까지만 깎음
        if cv2.countNonZero(coverage_mask) == 0:
            coverage_mask[dist_m >= self.robot_width] = 255
            
        return coverage_mask

    def _order_swaths(self, swath_pairs, entry_hint, exit_hint=None):
        """entry_hint(없으면 exit_hint 역산) 기준 스와스 정렬 - Step3
        인라인 로직과 _compute_node_raw_points가 공유하는 단일 구현.
        enable_entry_hint_ordering=False면 진입/진출 힌트를 모두 무시하고
        order_swaths_by_entry(swath_pairs, None)의 기본 순서(첫 스와스
        시작점 기준)를 강제함 - ablation 실험용 토글, HISTORY.md §2 참고."""
        if not self.enable_entry_hint_ordering:
            return geometry.order_swaths_by_entry(swath_pairs, None)
        if entry_hint is None:
            if exit_hint is not None:
                ordered_pairs = geometry.order_swaths_by_entry(swath_pairs, exit_hint)
                return [(p2, p1) for p1, p2 in reversed(ordered_pairs)]
            return geometry.order_swaths_by_entry(swath_pairs, None)
        return geometry.order_swaths_by_entry(swath_pairs, entry_hint)

    def _compute_node_raw_points(self, node_idx, entry_hint, exit_hint=None):
        """지정 노드의 F2C 커버리지 raw_points를 생성함(entry_hint 기준
        방향 정렬 - entry_hint가 None일 때만 exit_hint로 대신 정렬). Step3의
        인라인 계산과 정확히 동일한 로직임(디버그 이미지 저장 부분만
        제외) - _reorder_pendant_groups가 허브 노드의 실제 coverage 종료
        지점을 Step3보다 먼저 알아야 해서 이 부분만 별도 메서드로 추출함.
        node['bucket']/['safe_node_mask']가 이미 채워져 있어야 함(Step2
        완료 후에만 호출 가능)."""
        bucket = self.nodes[node_idx]['bucket']
        safe_node_mask = self.nodes[node_idx]['safe_node_mask']

        if bucket in ('narrow', 'ultra_narrow'):
            forced_angle = geometry.get_long_axis_angle_rad(safe_node_mask)
            swath_pairs = coverage.generate_raw_swaths(safe_node_mask, self.robot_params, decompose=True, split_angle_rad=forced_angle)
        else:
            swath_pairs = coverage.generate_raw_swaths(safe_node_mask, self.robot_params)

        raw_points = []
        if swath_pairs:
            ordered_pairs = self._order_swaths(swath_pairs, entry_hint, exit_hint)
            for p1, p2 in ordered_pairs:
                raw_points.extend([p1, p2])
        else:
            centroid = geometry.get_centroid(self.nodes[node_idx]['driveable_mask'])
            if centroid:
                raw_points.append(centroid)
        return raw_points

    def _compute_repass_adjusted_exit(self, raw_points):
        """coverage 노드 하나(raw_points)의 실제 물리적 exit 지점 - F2C
        스와스 자체의 마지막 점(raw_points[-1])이 아니라, 실행 시
        BoundaryRepassController.run_exit_repass가 되짚기를 마친 뒤 로봇이
        실제로 서 있게 될 위치(retrace 지점)를 반환함.

        boundary_repass.py의 _repass_distance_m/_offset_pose와 완전히
        동일한 기하 규칙을 따름: 마지막 다리(raw_points[-2:])의 heading을
        구하고, 왕복 거리는 설정값(boundary_repass_distance_m)과 그 다리
        길이의 90% 중 작은 쪽으로 clamp한 뒤, 그만큼 되짚어 물러난 지점을
        계산함. enable_boundary_repass가 꺼져 있거나, 다리가 없거나(단일
        점 방), clamp된 거리가 0.3m 미만이면(run_exit_repass 자신도 이 경우
        되짚기 없이 즉시 캡처를 끄므로) 원래 F2C 종료 지점을 그대로
        반환함 - 그 경우 로봇은 실제로 거기 그대로 있기 때문임.

        Step3의 current_pos(다음 노드로 가는 transit A*의 실제 시작점)와
        _reorder_pendant_groups의 허브 anchor 양쪽에서 재사용함 - 로봇이
        실제로 그 위치에서 다음 이동을 시작하므로, 오프라인 계획(및 그
        시각화)도 거기서부터 transit을 그려야 실제 주행과 일치함."""
        if not raw_points:
            return None
        if not self.enable_boundary_repass or len(raw_points) < 2:
            return raw_points[-1]

        a = np.array(raw_points[-2], dtype=float)
        b = np.array(raw_points[-1], dtype=float)
        vec = b - a
        seg_len_px = float(np.hypot(vec[0], vec[1]))
        if seg_len_px < 1e-6:
            return raw_points[-1]

        d_px = self.boundary_repass_distance_m / self.map_resolution
        d = min(d_px, seg_len_px * 0.9)
        if d * self.map_resolution < 0.3:
            return raw_points[-1]

        unit = vec / seg_len_px
        retrace = b - unit * d
        return (int(round(retrace[0])), int(round(retrace[1])))

    def _reorder_pendant_groups(self, tsp_sequence, detailed_sequence, node_waypoints):
        """
        Christofides 근사(tsp.py)는 F2C 스와스가 생성되기 전, 노드
        중심점(centroid) 거리만으로 방문 순서를 정함. 허브형 토폴로지
        (중앙 복도 하나에 여러 방이 매달린 구조, 예: Apt.dae의 node2 <->
        {1,4,5,6})에서는 각 pendant 노드의 실제 연결 지점(waypoint)이
        허브의 실제 coverage 종료 지점(entry_hint에 의해서만 정해짐 -
        exit_hint는 고려하지 않음, order_swaths_by_entry 참고)에서 얼마나
        가까운지를 이 근사가 전혀 반영하지 못함(발견 경위·실측 결과는
        HISTORY.md §2 참고).

        허브 h의 coverage가 확정된 직후(=h의 실제 물리적 exit 좌표를 알 수
        있는 시점), h에 '직접' 연결된(다른 노드를 거치지 않는) pendant
        노드들이 tsp_sequence 상에서 연속으로 나타나는 구간(=서로 직접
        연결되어 있지 않아 매번 h를 되짚어 지나가야만 하는 구간)을 찾아,
        그 구간만 h의 실제 exit 좌표에서부터 시작하는 nearest-neighbor
        순서로 재배열함. Christofides가 정한 전역적인 큰 흐름(어느
        허브/군집을 먼저·나중에 방문할지)은 건드리지 않고, 이미 정해진
        허브 도착 이후의 로컬 pendant 방문 순서만 다듬는 국소적 후처리임.

        pendant 사이의 이동 비용은 각자의 실제 coverage 스와스 형태까지
        고려하지 않고, hub<->pendant 연결 지점(waypoint) 사이의 유클리드
        거리로 근사함 - 매번 hub 복도를 되짚어 지나가야 하는 이 특정
        상황에서는 '문이 서로 얼마나 가까운가'가 실제 이동거리를 잘
        근사하기 때문임(각 pendant 자체의 coverage 왕복 비용은 방문
        순서와 무관하게 고정되므로 비교 대상에서 제외해도 됨).

        detailed_sequence 구조가 예상(hub와 pendant가 정확히 번갈아 나오는
        패턴)과 다르면(예: pendant끼리 직접 연결되어 있어 Dijkstra가 hub를
        경유하지 않은 경우) 안전하게 해당 구간의 재배열을 건너뜀 - 잘못된
        가정으로 경로 데이터를 조용히 훼손하는 것보다 나음.
        """
        tsp_sequence = list(tsp_sequence)
        detailed_sequence = list(detailed_sequence)

        def _dist(a, b):
            return float(np.hypot(a[0] - b[0], a[1] - b[1]))

        # tsp_sequence[j]가 detailed_sequence의 어느 인덱스에서 '커버리지
        # 방문'으로 등장하는지 Step3와 동일한 매칭 규칙으로 미리 기록해둠.
        target_positions = {}
        tsp_idx = 0
        for idx, n in enumerate(detailed_sequence):
            if tsp_idx < len(tsp_sequence) and n == tsp_sequence[tsp_idx]:
                target_positions[tsp_idx] = idx
                tsp_idx += 1

        j = 0
        while j < len(tsp_sequence):
            hub = tsp_sequence[j]
            run_start = j + 1
            k = run_start
            while k < len(tsp_sequence) and (hub, tsp_sequence[k]) in node_waypoints:
                k += 1
            run = tsp_sequence[run_start:k]

            if len(run) >= 2:
                hub_pos = target_positions.get(j)
                if hub_pos is not None:
                    expected_old = []
                    for idx2, n in enumerate(run):
                        expected_old.append(n)
                        if idx2 != len(run) - 1:
                            expected_old.append(hub)
                    seg_start = hub_pos + 1
                    seg_end = seg_start + len(expected_old)
                    old_slice = detailed_sequence[seg_start:seg_end]

                    if old_slice == expected_old:
                        prev_node = detailed_sequence[hub_pos - 1] if hub_pos > 0 else None
                        hub_entry_hint = node_waypoints.get((prev_node, hub)) if prev_node is not None else None
                        hub_raw_points = self._compute_node_raw_points(hub, hub_entry_hint)
                        # 허브 자신도 exit repass를 거치므로, pendant 순서를
                        # 정하는 anchor는 F2C 종료 지점이 아니라 repass가
                        # 끝난 뒤 로봇이 실제로 서 있을 위치여야 함
                        # (_compute_repass_adjusted_exit 참고) - Step3의
                        # current_pos 갱신과 동일한 기준.
                        anchor = self._compute_repass_adjusted_exit(hub_raw_points) or \
                            geometry.get_centroid(self.nodes[hub]['driveable_mask'])

                        if anchor is not None:
                            remaining = list(run)
                            new_order = []
                            pos = anchor
                            while remaining:
                                best = min(remaining, key=lambda n: _dist(pos, node_waypoints[(hub, n)]))
                                new_order.append(best)
                                pos = node_waypoints[(hub, best)]
                                remaining.remove(best)

                            if new_order != run:
                                print(f"[*] Pendant reorder: hub Node {hub + 1}'s neighbors "
                                      f"{[n + 1 for n in run]} -> {[n + 1 for n in new_order]} "
                                      f"(nearest-neighbor from hub's actual coverage exit point).")
                                tsp_sequence[run_start:k] = new_order
                                new_slice = []
                                for idx2, n in enumerate(new_order):
                                    new_slice.append(n)
                                    if idx2 != len(new_order) - 1:
                                        new_slice.append(hub)
                                detailed_sequence[seg_start:seg_end] = new_slice
                    else:
                        print(f"[WARN] Pendant reorder for hub Node {hub + 1}: detailed_sequence "
                              f"구조가 예상과 달라(pendant끼리 직접 연결된 경우 등) 재배열을 건너뜁니다.")

            j = k if len(run) >= 2 else run_start

        return tsp_sequence, detailed_sequence

    def execute_full_mission(self):
        """하위 알고리즘 모듈들을 오케스트레이션해서 전체 로봇 mission plan을 생성함."""
        print(f"\n{'-'*14} [Full Mission Planning Start] {'-'*14}")
        start_time = time.time()
        
        # [Step 1] TSP 순서 및 상세 경유 시퀀스 계산
        print("[Step 1/3] Calculating TSP and transit sequence...")
        tsp_sequence, detailed_sequence, planning_mask, node_waypoints, _connection_widths_px, _connection_masks = tsp.solve_tsp_sequence(
            nodes=self.nodes, global_mask=self.global_mask
        )
        
        # Transit A* 전용 글로벌 비용 지도(Cost Map) 생성
        print("[*] Generating Safe Cost Map for Transit Paths...")
        global_cost_map = self.generate_cost_map(self.global_mask)
        
        # [Step 2] 각 타겟 노드별 F2C 측정(Coverage) 경로 계산
        print("[Step 2/3] Generating F2C coverage paths...")
        width_bucket_counts = {'wide': 0, 'narrow': 0, 'ultra_narrow': 0}
        width_bucket_log = []  # (node_id, safe_width_px, bucket) - 필요 시 CSV로 dump 가능

        for node_idx in range(len(self.nodes)):
            # 원본 도면이 아닌 마진이 확보된 Coverage 전용 마스크 전달
            safe_node_mask = self.generate_coverage_mask(self.nodes[node_idx]['driveable_mask'])
            safe_width_px = geometry.estimate_min_width_px(safe_node_mask)

            node_id = self.nodes[node_idx]['id']
            contours, _ = cv2.findContours(safe_node_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if node_id == 2:
                print(f"[DEBUG] Node {node_id}의 safe_node_mask 윤곽선 개수: {len(contours)}")
                if len(contours) > 1:
                    areas = [cv2.contourArea(c) for c in contours]
                    print(f"[DEBUG] 각 조각의 면적: {areas}")

            if contours:
                c = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(c)
                box_pts = cv2.boxPoints(rect).astype(int)

                debug_img = cv2.cvtColor(self.nodes[node_idx]['driveable_mask'], cv2.COLOR_GRAY2BGR)
                cv2.drawContours(debug_img, [box_pts], 0, (0, 0, 255), 2)
                cv2.drawContours(debug_img, [c], -1, (0, 255, 0), 1)
                cv2.putText(debug_img, f"node {node_id}: {safe_width_px:.1f}px", (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                width_debug_dir = os.path.join(self.visualization_dir, "width_debug")
                os.makedirs(width_debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(width_debug_dir, f"node_{node_id:03d}.png"), debug_img)

            assist_needed = safe_width_px < self.blind_radius_px # bucket 분류용으로만 사용
            bucket = ('ultra_narrow' if assist_needed
                    else 'narrow' if safe_width_px < 2 * self.blind_radius_px
                    else 'wide')
            width_bucket_counts[bucket] += 1
            width_bucket_log.append((node_id, round(safe_width_px, 1), bucket))

            self.nodes[node_idx]['bucket'] = bucket
            self.nodes[node_idx]['safe_node_mask'] = safe_node_mask

        # 너비 측정 로그
        print(f"\n[*] Node width classification (blind_radius_px={self.blind_radius_px:.1f}):")
        print(f"    wide={width_bucket_counts['wide']}, narrow={width_bucket_counts['narrow']}, "
            f"ultra_narrow={width_bucket_counts['ultra_narrow']}  (total={len(width_bucket_log)})")
        for node_id, w, bucket in sorted(width_bucket_log, key=lambda t: t[1]):
            print(f"      node {node_id:>3}: safe_width_px={w:>6}  -> {bucket}")

        # [Step 2.5] 허브형 토폴로지(중앙 복도 하나에 여러 방이 매달린 구조)의
        # pendant 방문 순서를 허브의 실제 coverage 종료 지점 기준으로 재정렬.
        # bucket/safe_node_mask가 막 채워진 직후라야 각 노드의 F2C 스와스를
        # 생성할 수 있어 여기(Step2 이후, Step3 이전)에서 수행함. 자세한
        # 이유는 _reorder_pendant_groups 참고.
        # ablation 토글: enable_pendant_reorder=False면 Christofides 근사가
        # 정한 순서를 그대로 두고 이 국소 재정렬을 건너뜀 (HISTORY.md §2 참고).
        if self.enable_pendant_reorder:
            tsp_sequence, detailed_sequence = self._reorder_pendant_groups(
                tsp_sequence, detailed_sequence, node_waypoints
            )

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib.patches import Patch

            os.makedirs(self.visualization_dir, exist_ok=True)

            sorted_log = sorted(width_bucket_log, key=lambda t: t[1])
            node_labels = [f"node {nid}" for nid, _, _ in sorted_log]
            widths = [w for _, w, _ in sorted_log]
            buckets = [b for _, _, b in sorted_log]

            bucket_colors = {'wide': '#2ca02c', 'narrow': '#ff9900', 'ultra_narrow': '#d62728'}
            bar_colors = [bucket_colors[b] for b in buckets]

            fig, ax = plt.subplots(figsize=(max(8, len(sorted_log) * 0.9), 6))
            bars = ax.bar(range(len(sorted_log)), widths, color=bar_colors, edgecolor='black')

            for bar, w in zip(bars, widths):
                ax.text(bar.get_x() + bar.get_width() / 2, w + max(widths) * 0.015,
                        f"{w:.1f}", ha='center', va='bottom', fontsize=9)

            ax.set_xticks(range(len(sorted_log)))
            ax.set_xticklabels(node_labels, rotation=45, ha='right')
            ax.axhline(self.blind_radius_px, color='orange', linestyle='--',
                    label=f'blind_radius_px ({self.blind_radius_px:.0f})')
            ax.axhline(2 * self.blind_radius_px, color='red', linestyle='--',
                    label=f'2x blind_radius_px ({2 * self.blind_radius_px:.0f})')
            ax.set_ylabel('safe_width_px')
            ax.set_title('Node width classification (per-node)')

            bucket_handles = [Patch(facecolor=bucket_colors[b], edgecolor='black', label=b)
                            for b in ['wide', 'narrow', 'ultra_narrow']]
            line_handles, line_labels = ax.get_legend_handles_labels()
            ax.legend(handles=bucket_handles + line_handles, loc='upper left')

            summary_line = (f"wide={width_bucket_counts['wide']}, narrow={width_bucket_counts['narrow']}, "
                            f"ultra_narrow={width_bucket_counts['ultra_narrow']}  (total={len(width_bucket_log)})")
            fig.suptitle(summary_line, fontsize=10, y=0.98)

            detail_lines = [f"node {nid:>3}: safe_width_px={w:>6} -> {b}" for nid, w, b in sorted_log]
            fig.text(0.02, -0.02, "\n".join(detail_lines), fontsize=7, family='monospace', va='top')

            plt.tight_layout(rect=[0, 0.02, 1, 0.95])
            hist_path = os.path.join(self.visualization_dir, 'width_classification_histogram.png')
            plt.savefig(hist_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[*] Histogram saved to: {hist_path}")
        except ImportError:
            print("[WARN] matplotlib not available - counts above are still printed.")

        
        # [Step 3] 최종 궤적 생성 및 연결 (Transit via A*)
        print("[Step 3/3] Finalizing trajectory (Linking all paths)...")
        self.path_segments = [] 
        current_pos = None
        tsp_idx = 0 
        
        coverage_count = 0
        transit_count = 0

        def simplify_path(path, epsilon_px=3.0):
            """
            A* 8방향 격자 이동이 만드는 계단식(staircase) 지그재그를 제거하고 진짜
            꺾이는 지점만 남김. 격자는 임의 각도의 직선을 정확히 못 그리고 두
            방향을 번갈아 밟아 근사하는데, 이 계단 하나하나를 sampler.py의 앵커
            감지 로직이 '진짜 코너'로 착각하는 문제를 막기 위함임.
            """
            if len(path) < 3:
                return path
            arr = np.array(path, dtype=np.int32).reshape((-1, 1, 2))
            simplified = cv2.approxPolyDP(arr, epsilon_px, closed=False)
            return [tuple(map(int, pt[0])) for pt in simplified]

        for i in range(len(detailed_sequence)):
            curr_node = detailed_sequence[i]
            
            # 1. 측정(Coverage) 대상 노드 처리
            if tsp_idx < len(tsp_sequence) and curr_node == tsp_sequence[tsp_idx]:
                prev_node = detailed_sequence[i - 1] if i > 0 else None
                entry_hint = node_waypoints.get((prev_node, curr_node)) if prev_node is not None else None

                bucket = self.nodes[curr_node]['bucket']
                safe_node_mask = self.nodes[curr_node]['safe_node_mask']

                if bucket in ('narrow', 'ultra_narrow'):
                    forced_angle = geometry.get_long_axis_angle_rad(safe_node_mask)
                    swath_pairs = coverage.generate_raw_swaths(safe_node_mask, self.robot_params, decompose=True, split_angle_rad=forced_angle)
                else:
                    forced_angle = None
                    swath_pairs = coverage.generate_raw_swaths(safe_node_mask, self.robot_params)

                if self.nodes[curr_node]['id'] in (1, 2):
                    node_id_dbg = self.nodes[curr_node]['id']
                    angle_str = f"{math.degrees(forced_angle):.1f}" if forced_angle is not None else "N/A(wide)"
                    print(f"[DEBUG] Node {node_id_dbg}: forced_angle_deg={angle_str}, "
                        f"num_swaths={len(swath_pairs)}, swath_pairs={swath_pairs}")

                    debug_img = cv2.cvtColor(safe_node_mask, cv2.COLOR_GRAY2BGR)
                    contours, _ = cv2.findContours(safe_node_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 1)  # 초록: 폴리곤 전체
                    for p1, p2 in swath_pairs:
                        cv2.line(debug_img, p1, p2, (0, 0, 255), 2)   # 빨강: 실제 생성된 스와스
                        cv2.circle(debug_img, p1, 4, (255, 0, 0), -1)  # 파랑: 스와스 시작점
                        cv2.circle(debug_img, p2, 4, (0, 255, 255), -1)  # 노랑: 스와스 끝점

                    dbg_dir = os.path.join(self.visualization_dir, "width_debug")
                    os.makedirs(dbg_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(dbg_dir, f"node_{node_id_dbg:03d}_swaths.png"), debug_img)  # 파일명에 id 반영
                    print(f"[DEBUG] Saved node_{node_id_dbg:03d}_swaths.png")

                raw_points = []
                if swath_pairs:
                    # 미션의 첫 coverage 노드는 진입 기준점이 없음 - 대신 다음 노드로
                    # 나가는 출구 방향에 최대한 가깝게 '끝나도록' exit_hint를 역산해
                    # 넘김(_order_swaths가 통째로 뒤집어 처리, HISTORY.md §2 참고).
                    exit_hint = None
                    if entry_hint is None:
                        next_node = detailed_sequence[i + 1] if i < len(detailed_sequence) - 1 else None
                        exit_hint = node_waypoints.get((curr_node, next_node)) if next_node is not None else None

                    ordered_pairs = self._order_swaths(swath_pairs, entry_hint, exit_hint)

                    for p1, p2 in ordered_pairs:
                        raw_points.extend([p1, p2])
                else:
                    centroid = geometry.get_centroid(self.nodes[curr_node]['driveable_mask'])
                    if centroid: raw_points.append(centroid)
                
                if raw_points:
                    # 진입/진출 힌트 확보: detailed_sequence 상에서 이 노드의 바로 앞/뒤
                    # 노드와의 연결 지점(waypoints, tsp.py의 extract_waypoints가 계산한
                    # '문지방 등 안전 통과 지점'). 두 노드가 항상 그래프 상 인접하도록
                    # detailed_sequence가 Dijkstra 최단경로로 구성되어 있으므로, 이
                    # 조회는 항상 유효한 값을 반환함(첫/마지막 노드의 바깥쪽 방향 제외).

                    # 방향 최적화: 진입점 근접성뿐 아니라 진출점(다음 노드로 가는 방향)까지
                    # 함께 고려해서 정방향/역방향을 선택함. 이렇게 해야 예를 들어 방 A ->
                    # 복도 B -> 복도 C로 이동할 때, B의 coverage가 A쪽에서 들어와서 C쪽으로
                    # 나가도록 자연스럽게 정렬되어 불필요한 되돌아가기(우회 transit)가 줄어듦.

                    # 노드 진입 경로 (Transit) 계산 (A* 알고리즘)
                    if current_pos:
                        via_point = entry_hint  # curr_node로 들어가는 연결부의 로컬 중심점
                        enter_path = []

                        if via_point is not None and via_point != current_pos:
                            print(f"[TRACE_ENTER] Leg 1 (via doorway center): {current_pos} -> {via_point}")
                            _, leg1 = transit.find_path_with_penalty(
                                start=current_pos, goal=via_point, planning_mask=global_cost_map,
                                turn_weight=self.turn_weight, wall_weight=self.wall_weight
                            )
                            if leg1:
                                enter_path.extend(leg1)
                            else:
                                print(f"[WARN] Leg1 (doorway center 경유) 실패 - 직접 경로로 폴백.")

                        leg2_start = enter_path[-1] if enter_path else current_pos
                        print(f"[TRACE_ENTER] Leg 2: {leg2_start} -> {raw_points[0]}")
                        _, leg2 = transit.find_path_with_penalty(
                            start=leg2_start, goal=raw_points[0], planning_mask=global_cost_map,
                            turn_weight=self.turn_weight, wall_weight=self.wall_weight
                        )

                        if leg2:
                            if enter_path and enter_path[-1] == leg2[0]:
                                full_enter_path = enter_path + leg2[1:]
                            else:
                                full_enter_path = enter_path + leg2

                            if self.enable_path_simplification:
                                full_enter_path = simplify_path(full_enter_path, epsilon_px=3.0)  # A* 지그재그 제거는 유지
                            self.path_segments.append({'type': 'transit', 'path': full_enter_path, 'record_pcd': False})
                            transit_count += 1
                        else:
                            print(f"[ERROR] Cannot find safe path to Node {curr_node+1}. Wall detected!")
                    
                    # 측정 경로 추가 - coverage 자체는 F2C 스와스 그대로
                    # 저장함(raw_points, 종료 지점은 여전히 raw_points[-1]).
                    self.path_segments.append({'type': 'coverage', 'path': raw_points, 'record_pcd': True})
                    coverage_count += 1
                    # 다음 노드로 가는 transit(Leg1)은 F2C 종료 지점이 아니라
                    # exit repass가 끝난 뒤 로봇이 실제로 있을 위치에서
                    # 시작해야 함 - 안 그러면 오프라인 계획/시각화가 "repass가
                    # 없는 것처럼" coverage 끝점에서 곧장 transit이 이어지는
                    # 것으로 그려지는데, 실제로는 그 사이에 되짚기 왕복이 있음
                    # (도입 경위는 HISTORY.md §2 참고). repass_preview 화살표
                    # (_build_boundary_repass_preview)가 raw_points[-1]->이
                    # 지점 구간을 시각적으로 이어줌.
                    current_pos = self._compute_repass_adjusted_exit(raw_points)
                
                print(f"    -> Completed Coverage task in Node {curr_node+1}")
                tsp_idx += 1 
            else:
                print(f"    -> Transiting through Node {curr_node+1}")
            
        total_time = time.time() - start_time
        print(f"\n[DEBUG] Path Segments Created: Coverage({coverage_count}), Transit({transit_count})")
        print(f"{'-'*20} [Planning Completed in {total_time:.2f}s] {'-'*20}\n")

        return tsp_sequence

    def _build_boundary_repass_preview(self):
        """미션 실행 시 BoundaryRepassController가 만들 왕복 경로를 계획
        단계에서 근사해 시각화 전용으로 반환함. boundary_repass.py의 기하
        규칙(_offset_pose/_repass_distance_m)을 px 단위로 그대로 재현함.
        self.path_segments(=실제 final_path.json 원본)는 건드리지 않음.

        heading 계산 시 주의: self.path_segments의 'coverage' 항목 하나는
        F2C 스와스 전부(꺾이는 코너 포함)를 이어붙인 좌표 목록이라, 여러
        스와스가 꺾여있는 노드는 path[0]->path[-1] 전체 직선(코너 무시한
        거시적 방향)이 실제 로봇이 그 시작/끝 지점에서 나아가는 방향과 전혀
        다를 수 있음(발견 경위는 HISTORY.md §2 참고). mission_executor.py의
        실제 run_start_prepass/run_exit_repass는 heading 변화 기준으로
        분할된 sub-segment(첫/마지막 직선 다리 하나)만 넘겨받으므로 이 문제가
        없음 - 여기서도 동일하게 첫 다리(path[0]->path[1])/마지막 다리
        (path[-2]->path[-1])만으로 heading과 길이(clamp 기준)를 계산해서
        맞춤. 앵커 지점(p0/p_end) 자체는 그대로 path[0]/path[-1] 사용.
        """
        preview_segments = []
        if not self.path_segments:
            return preview_segments

        d_m = self.boundary_repass_distance_m
        d_px = d_m / self.map_resolution

        def _unit_and_len(path):
            p0 = np.array(path[0], dtype=float)
            p1 = np.array(path[-1], dtype=float)
            vec = p1 - p0
            length = float(np.hypot(vec[0], vec[1]))
            if length < 1e-6:
                return None, 0.0
            return vec / length, length

        print("[*] Boundary repass preview:")

        first_seg = self.path_segments[0]
        if first_seg['type'] == 'coverage' and len(first_seg['path']) >= 2:
            unit, seg_len_px = _unit_and_len(first_seg['path'][:2])  # 첫 다리만
            if unit is not None:
                d = min(d_px, seg_len_px * 0.9)
                p0 = np.array(first_seg['path'][0], dtype=float)
                runway = p0 + unit * d
                # 화살표는 실제 로봇 이동 순서(runway -> p0)를 나타내야 함 -
                # run_start_prepass는 로봇이 runway 지점에서 스폰되어 p0로
                # 들어가는 편도 주행이므로, coverage 진행 방향(p0->runway 방향인
                # unit)과는 반대 방향으로 그려야 맞음. exit repass 화살표(아래,
                # p_end->retrace)와 동일한 "모션 시작점->끝점" 관례를 따름.
                preview_segments.append({
                    'type': 'repass_preview',
                    'path': [tuple(runway.astype(int)), tuple(p0.astype(int))],
                    'label': f'START HERE ({d * self.map_resolution:.2f}m)',
                    'label_at': 0,  # runway 지점(실제 로봇을 놔야 하는 곳)에 라벨을 붙임 -
                                    # p0는 이미 순번 "1"이 찍혀있어서 그쪽에 붙이면 안 보임.
                })
                print(f"    [start prepass] mission start (path_segments[0]) - "
                      f"runway {d * self.map_resolution:.2f}m")
            else:
                print("    [start prepass] SKIPPED - start coverage segment has zero length.")
        else:
            print("    [start prepass] SKIPPED - path_segments[0] is not type='coverage'.")

        n_coverage_exits = 0
        n_previewed = 0
        for idx, seg in enumerate(self.path_segments):
            if seg['type'] != 'coverage':
                continue
            n_coverage_exits += 1
            is_mission_end = (idx == len(self.path_segments) - 1)
            tag = f"coverage exit #{n_coverage_exits} (path_segments[{idx}]" \
                  f"{', mission end' if is_mission_end else ''})"

            if len(seg['path']) < 2:
                print(f"    [exit repass] {tag} SKIPPED - single-point coverage, no heading to retrace along.")
                continue

            # _compute_repass_adjusted_exit와 완전히 동일한 계산을 재사용함
            # (Step3의 current_pos 갱신이 실제로 쓰는 바로 그 함수) - 이 함수와
            # 별개로 공식을 중복 구현하면 enable_boundary_repass=False일 때도
            # 그 사실을 모른 채 무조건 화살표를 그리는 불일치가 생기므로 통일함.
            # self.path_segments 자체가 이미 이 지점에서 시작하므로, 여기서는
            # "coverage 끝점 -> 그 실제 시작점" 구간만 시각적으로 이어주면 됨.
            p_end = np.array(seg['path'][-1], dtype=float)
            retrace_raw = self._compute_repass_adjusted_exit(seg['path'])
            if retrace_raw is None or tuple(retrace_raw) == tuple(seg['path'][-1]):
                print(f"    [exit repass] {tag} SKIPPED - no repass applied "
                      f"(disabled, or clamped distance too short).")
                continue

            retrace = np.array(retrace_raw, dtype=float)
            d_m = float(np.hypot(*(p_end - retrace))) * self.map_resolution
            preview_segments.append({
                'type': 'repass_preview',
                'path': [tuple(p_end.astype(int)), tuple(retrace.astype(int))],
                'label': f'exit repass #{n_coverage_exits} ({d_m:.2f}m)',
                'label_at': 1,  # retrace 지점(되짚어 나가야 하는 곳)에 라벨
            })
            n_previewed += 1
            print(f"    [exit repass] {tag} - retrace {d_m:.2f}m")

        print(f"[*] Boundary repass preview summary: {n_previewed}/{n_coverage_exits} "
              f"coverage exits got a repass preview (rest skipped as logged above).")

        return preview_segments

    def plan(self, save_debug=True, show_plot=False, output_dir=None):
        # output_dir 미지정 시, 현재 작업 디렉토리(cwd)에 의존하는 상대경로
        # "analytics/metrics" 대신 외부 저장소(workspace_root) 기준 절대경로로 fallback.
        if output_dir is None:
            default_workspace_root = os.path.expanduser("~/dae_floor_maps")
            output_dir = os.path.join(default_workspace_root, "analytics/metrics")
            print(f"[WARN] 'output_dir' not provided to plan(). Falling back to: {output_dir}")

        # 1. 전역 미션 계획 실행 (Pixel 단위 경로 생성)
        self.execute_full_mission()

        # 2. 경로 생성 실패 시 예외 처리
        if not self.path_segments:
            print("[WARN] No path generated. Mission aborted.")
            return None
        
        # 3. 결과 시각화 (Visualizer)
        # boundary_repass 미리보기는 시각화 전용 목록에만 추가 - self.path_segments
        # 자체(translator로 넘어가 final_path.json이 되는 원본)는 그대로 둠.
        viz_path_segments = self.path_segments + self._build_boundary_repass_preview()

        if save_debug:
            print(f"[*] Saving debug visualization to centralized storage...")
            os.makedirs(self.visualization_dir, exist_ok=True)
            visualizer.save_debug_image(
                nodes=self.nodes,
                path_segments=viz_path_segments,
                global_mask=self.global_mask,
                output_dir=self.visualization_dir,
                filename="full_mission_path.png"
            ) # full_mission_path.png 저장

        if show_plot:
            print("[*] Displaying mission state on screen.")
            visualizer.plot_mission_state(
                nodes=self.nodes,
                path_segments=viz_path_segments,
                global_mask=self.global_mask
            )

        # 4. Translator를 통한 좌표 변환 및 메시지 포맷팅 (Pixel -> Meter)
        # Y축 대칭 반전 역산을 위해 전역 마스크 이미지의 세로 픽셀 크기(Height)를 추출함.
        map_height = self.global_mask.shape[0]

        print("[*] Translating path segments to Metric coordinates...")

        raw_nav2_path = translator.convert_segments_to_nav2(
            path_segments=self.path_segments,
            origin=self.origin,
            resolution=self.map_resolution,
            map_height=map_height
        )

        # sampling_step만큼의 거리마다 샘플링
        sampled_nav2_path = sampler.interpolate_with_semantics(
            raw_nav2_path
        )
        
        raw_flat_path = []
        for seg in raw_nav2_path:
            for p in seg['poses']:
                p_copy = json.loads(json.dumps(p))
                p_copy['header'] = {
                    'frame_id': 'map',
                    'task_type': seg['type'],
                    'record_pcd': seg.get('record_pcd', seg['type'] == 'coverage'),
                }
                x, y = p_copy['pose']['position']['x'], p_copy['pose']['position']['y']
                
                # 거리 기반 비교
                if not raw_flat_path or math.hypot(raw_flat_path[-1]['pose']['position']['x'] - x,
                                                raw_flat_path[-1]['pose']['position']['y'] - y) > 0.001:
                    raw_flat_path.append(p_copy)

        os.makedirs(output_dir, exist_ok=True)
        raw_output_file = os.path.join(output_dir, "raw_path.json")
        sampled_output_file = os.path.join(output_dir, "final_path.json")

        with open(raw_output_file, 'w') as f:
            json.dump(raw_flat_path, f, indent=4)
            
        with open(sampled_output_file, 'w') as f:
            json.dump(sampled_nav2_path, f, indent=4)

        # final_path.json 자체가 이 값들(특히 boundary_repass_distance_m/
        # enable_boundary_repass, _compute_repass_adjusted_exit 참고)에
        # 기하학적으로 의존하므로, 계획 시점과 실행 시점(mission_executor.py가
        # params.yaml에서 직접 읽음)의 값이 어긋나면 계획된 transit 시작점과
        # 실제 repass 후 로봇 위치가 조용히 달라짐 - 계획 시점에 실제로 쓴
        # 값을 사이드카 파일로 남겨서 mission_executor.py가 시작 시 자기
        # params.yaml 값과 자동 대조하게 함(다르면 다른 CRITICAL ERROR들과
        # 동일하게 즉시 중단 - _load_final_path 참고, 도입 경위는 HISTORY.md
        # §2 참고).
        meta_output_file = os.path.join(output_dir, "final_path_meta.json")
        plan_meta = {
            'robot_width': self.robot_width,
            'path_safety_margin': self.path_safety_margin,
            'boundary_repass_distance_m': self.boundary_repass_distance_m,
            'enable_boundary_repass': self.enable_boundary_repass,
            'map_resolution': self.map_resolution,
            'blind_radius_m': self.blind_radius_m,
            # 아래 3개는 실행 시 참조/대조되지 않음(순수 계획 단계 좌표
            # 생성에만 관여) - ablation 실험 시 이 final_path.json이 어떤
            # 토글 조합으로 생성됐는지 추적하기 위한 기록용 메타데이터.
            'enable_pendant_reorder': self.enable_pendant_reorder,
            'enable_entry_hint_ordering': self.enable_entry_hint_ordering,
            'enable_path_simplification': self.enable_path_simplification,
        }
        with open(meta_output_file, 'w') as f:
            json.dump(plan_meta, f, indent=4)

        print(f"[*] Mission Planner Successfully Completed.")
        print(f"    -> Raw Keypoints Path saved to: {raw_output_file} ({len(raw_flat_path)} pts)")
        print(f"    -> Sampled Path saved to: {sampled_output_file} ({len(sampled_nav2_path)} pts)")
        print(f"    -> Plan-time parameter snapshot saved to: {meta_output_file}")

        if save_debug:
            print("[*] Drawing path points on debug images...")
            
            # 1. 픽셀 좌표 변환 함수
            def get_pixel_points(pose_list_or_segments, is_raw=False):
                pts = []
                # 원본(raw)인 경우 중첩 리스트 구조, sampled된 경로인 경우 포인트들의 단일 리스트임.
                poses = []
                if is_raw:
                    for seg in pose_list_or_segments:
                        poses.extend(seg['poses'])
                else:
                    poses = pose_list_or_segments

                for p in poses:
                    mx = p['pose']['position']['x']
                    my = p['pose']['position']['y']
                    px = int((mx - self.origin[0]) / self.map_resolution)
                    py = int(map_height - (my - self.origin[1]) / self.map_resolution)
                    if 0 <= px < self.global_mask.shape[1] and 0 <= py < self.global_mask.shape[0]:
                        pts.append((px, py))
                return pts

            # 좌표 추출
            raw_pixel_points = get_pixel_points(raw_nav2_path, is_raw=True)
            sampled_pixel_points = get_pixel_points(sampled_nav2_path, is_raw=False)

            # 오버레이할 베이스 이미지 경로
            base_img_path = os.path.join(self.visualization_dir, "full_mission_path.png")

            # 2. 이미지 로드 및 오버레이
            def create_overlay_image(output_path, points):
                if os.path.exists(base_img_path):
                    img = cv2.imread(base_img_path)
                    if img is not None:
                        img = visualizer.draw_waypoint_on_image(img, points)
                        cv2.imwrite(output_path, img)
                        print(f"[*] Overlay saved to: {output_path}")
                    else:
                        print(f"[!] Failed to load base image: {base_img_path}")
                else:
                    print(f"[!] Base image not found: {base_img_path}")

            create_overlay_image(os.path.join(self.visualization_dir, "raw_waypoint.png"), raw_pixel_points)
            create_overlay_image(os.path.join(self.visualization_dir, "sampled_waypoint.png"), sampled_pixel_points)
            
        # 샘플링된 웨이포인트 반환. 필요하다면 원본(raw) 포인트를 반환해도 됨.
        return sampled_nav2_path
