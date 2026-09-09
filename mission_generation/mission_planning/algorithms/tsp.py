import numpy as np
import cv2
import networkx as nx

# 중심점 계산 유틸리티 
from mission_planning.utils import geometry

def extract_waypoints(nodes, global_mask):
    """
    노드 간의 연결 정보를 이용해 각 노드 간의 안전한 연결 지점(Waypoint)을 계산함.
    동시에 A* 탐색에서 좁은 경계를 안전하게 넘을 수 있도록 확장된 planning_mask를 반환함.
    """
    planning_mask = global_mask.copy()
    waypoints = {}
    connection_widths_px = {}   # (i,j)/(j,i) -> 연결부 로컬 통과 폭(px)
    connection_masks = {}       # (i,j)/(j,i) -> 해당 연결부의 dilated_overlap 마스크

    global_dist_px = cv2.distanceTransform(global_mask, cv2.DIST_L2, 5)
    
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            mask_i = nodes[i]['driveable_mask']
            mask_j = nodes[j]['driveable_mask']

            # 마스크 팽창을 통해 두 노드가 인접해 있는지 확인
            kernel = np.ones((3, 3), np.uint8)
            dilated_i = cv2.dilate(mask_i, kernel)
            overlap = cv2.bitwise_and(dilated_i, mask_j)
            
            if cv2.countNonZero(overlap) > 0:
                # overlap 위치에서 global_dist_px 값을 읽어와 그중 최댓값(=벽에서 가장 먼 안전한 지점)을 찾음.
                masked_dist = np.where(overlap > 0, global_dist_px, 0).astype(np.float32)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(masked_dist)
                
                if max_val > 0: 
                    cx, cy = max_loc 
                    waypoints[(i, j)] = (cx, cy)
                    waypoints[(j, i)] = (cx, cy)
                    
                    # 노드 경계선에서 넓게 통과할 수 있도록 overlap 영역 팽창 후 planning_mask에 추가.
                    kernel_pass = np.ones((5, 5), np.uint8)
                    dilated_overlap = cv2.dilate(overlap, kernel_pass)
                    planning_mask = cv2.bitwise_or(planning_mask, dilated_overlap)

                    # distanceTransform의 max_val은 "벽에서 가장 먼 지점까지의 거리(반경)"이므로, 
                    # 실제 통과 가능한 폭은 그 2배로 근사함.
                    conn_width_px = 2.0 * max_val
                    connection_widths_px[(i, j)] = conn_width_px
                    connection_widths_px[(j, i)] = conn_width_px
                    connection_masks[(i, j)] = dilated_overlap
                    connection_masks[(j, i)] = dilated_overlap
                    
                    print(f"[DEBUG - TSP] Connected: Node {i+1} <-> Node {j+1} at Waypoint ({cx}, {cy})")
                    
    return waypoints, planning_mask, connection_widths_px, connection_masks

def solve_tsp_sequence(nodes, global_mask):
    """
    노드 간 연결망을 분석해 최적의 방문 순서(TSP Sequence)와 상세 이동 경로를 계산함.

    Returns:
        tsp_path (list): 실제 Coverage를 수행해야 하는 노드들의 방문 순서
        detailed_sequence (list): 타겟으로 이동하기 위해 거쳐가는 경유 노드가 포함된 상세 시퀀스
        planning_mask (np.ndarray): A* 경로 탐색에 사용될 확장된 맵 마스크
        waypoints (dict): (i, j) 노드 쌍 -> (px, py) 연결 지점(문지방 등 안전 통과 지점).
            mission_planner.py가 각 노드의 coverage 진입/진출 방향을 TSP 방문 순서의
            흐름에 맞게 정하는 데 사용함(예: 방 A -> 복도 B -> 복도 C로 이동할 때,
            A의 coverage 종료 지점과 B의 coverage 시작 지점이 A-B 연결 지점 근처가
            되도록 정렬).

        connection_widths_px (dict): (i, j) 노드 쌍 -> 연결부 로컬 통과 폭(px).
        connection_masks (dict): (i, j) 노드 쌍 -> 해당 연결부의 dilated_overlap 마스크.
            둘 다 노드 자체는 넓어도 연결부(문지방 등)가 좁은 경우를 보완(assist_mask)하기
            위해 도입됐던 값인데, 그 보완 로직 자체가 실측 검증 후 불필요함이 확인되어
            제거됨(HISTORY.md §2 참고) - mission_planner.py는 현재 이 두 값을
            `_connection_widths_px`/`_connection_masks`로 언패킹만 하고 쓰지 않음.
    """
    # 1. 연결 지점 및 경로 탐색용 마스크 획득
    waypoints, planning_mask, connection_widths_px, connection_masks = extract_waypoints(nodes, global_mask)

    # 노드가 1개일 때는 tsp 건너뜀.
    if len(nodes) <= 1:
        tsp_path = [0] if len(nodes) == 1 else []
        detailed_sequence = [0] if len(nodes) == 1 else []

        print(f"\n[TSP Skipped] Only {len(nodes)} node(s). No TSP required.")

        return tsp_path, detailed_sequence, planning_mask, waypoints, connection_widths_px, connection_masks
    
    # 2. 노드 간 Metric Closure Graph 구축
    graph = nx.Graph()
    graph.add_nodes_from(range(len(nodes)))
    
    for (u, v), wp in waypoints.items():
        if not graph.has_edge(u, v):
            # core.py에 구현될 utils.geometry.get_centroid 활용
            c_u = geometry.get_centroid(nodes[u]['driveable_mask'])
            c_v = geometry.get_centroid(nodes[v]['driveable_mask'])
            
            if c_u and c_v:
                # 유클리드 거리를 가중치로 사용
                approx_cost = np.hypot(c_u[0] - c_v[0], c_u[1] - c_v[1])
                graph.add_edge(u, v, weight=approx_cost)
                
    # 3. 모든 노드 쌍에 대한 최단 경로 및 비용 계산 (Dijkstra)
    shortest_paths = dict(nx.all_pairs_dijkstra_path(graph, weight='weight'))
    shortest_path_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight='weight'))
    
    # TSP 알고리즘 적용을 위해 모든 노드가 연결된 Complete Graph 구성
    complete_graph = nx.complete_graph(len(nodes))
    for u in complete_graph.nodes():
        for v in complete_graph.nodes():
            if u != v:
                complete_graph[u][v]['weight'] = shortest_path_lengths[u].get(v, float('inf'))
                
    # 4. TSP 근사 알고리즘을 통한 Target 방문 스케줄링
    tsp_path = nx.approximation.traveling_salesman_problem(complete_graph, weight='weight', cycle=False)
    print(f"\n[TSP Target] Schedule: {[x+1 for x in tsp_path]}")
    
    # 5. 경유 노드를 포함한 상세 경로(Detailed Sequence) 결합
    detailed_sequence = []
    for i in range(len(tsp_path) - 1):
        u = tsp_path[i]
        v = tsp_path[i+1]
        if v in shortest_paths.get(u, {}):
            path_uv = shortest_paths[u][v]
            
            # 마지막 구간이 아니면 끝 노드를 제외하여 리스트 중복 삽입 방지
            if i == len(tsp_path) - 2:
                detailed_sequence.extend(path_uv)
            else:
                detailed_sequence.extend(path_uv[:-1])
                
    print(f"[Actual Routing] Detailed Sequence with Transits: {[x+1 for x in detailed_sequence]}")
    
    return tsp_path, detailed_sequence, planning_mask, waypoints, connection_widths_px, connection_masks
