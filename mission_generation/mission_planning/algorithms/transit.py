import heapq
import numpy as np

def find_path_with_penalty(start, goal, planning_mask, turn_weight=2, wall_weight=6.0):
    """
    장애물을 우회하며 회전 동작에 페널티를 부여하는 개선된 A* 탐색 알고리즘임.
    
    Args:
        start (tuple): (x, y) 시작 좌표
        goal (tuple): (x, y) 목표 좌표
        planning_mask (np.ndarray): 이동 가능한 영역이 255/1 등으로 표시된 확장 마스크
        turn_weight (int, optional): 회전 시 부과할 페널티 가중치. 높을수록 직진 선호.
        
    Returns:
        g_score (float): 총 경로 비용 (도달 불가 시 float('inf'))
        path (list): (x, y) 좌표 튜플의 리스트
    """
    # 1. 예외 처리: 시작/종료 지점이 장애물 내부인지 검사
    if planning_mask[start[1], start[0]] == 0 or planning_mask[goal[1], goal[0]] == 0:
        return float('inf'), []

    # 상태 정의: (x, y, 이전 이동 방향 dx, 이전 이동 방향 dy)
    start_state = (start[0], start[1], 0, 0)
    
    # 우선순위 큐 (Priority Queue): (f_score, g_score, x, y, pdx, pdy)
    pq = [(0, 0, start[0], start[1], 0, 0)]
    
    # 특정 상태까지 도달하는 최소 비용 기록 딕셔너리
    best_g_scores = {start_state: 0}
    
    # 경로 역추적을 위한 딕셔너리 (자식 상태 -> 부모 상태)
    came_from = {}
    
    # 8방향 이동 (dx, dy, step_cost)
    directions = [(0, -1, 1), (0, 1, 1), (-1, 0, 1), (1, 0, 1),
                  (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414)]

    while pq:
        f_score, g_score, x, y, pdx, pdy = heapq.heappop(pq)
        state = (x, y, pdx, pdy)
        
        # 더 나은 경로를 이미 찾은 경우 스킵
        if g_score > best_g_scores.get(state, float('inf')):
            continue
            
        # 2. 목표 도달 시 경로 역추적 (Backtracking)
        if (x, y) == goal:
            path = []
            curr_state = state
            while curr_state in came_from:
                path.append((curr_state[0], curr_state[1]))
                curr_state = came_from[curr_state]
            path.append((start[0], start[1])) # 시작점 추가
            path.reverse() # Start -> Goal 순서로 뒤집기
            return g_score, path
            
        # 3. 인접 노드 탐색
        for dx, dy, step_cost in directions:
            nx_x, nx_y = x + dx, y + dy
            
            # 맵 범위 내 존재 및 장애물 여부 검사
            if 0 <= nx_x < planning_mask.shape[1] and 0 <= nx_y < planning_mask.shape[0]:
                safety_score = planning_mask[nx_y, nx_x]

                if safety_score > 0: # 장애물이 아닐 때

                    # safety_score가 255(안전)이면 페널티 0, 50(위험)이면 높은 페널티 부여
                    # cost_map의 최솟값이 50이므로 분모를 205(255-50)로 하여 정규화
                    normalized_danger = (255 - safety_score) / 205.0 
                    wall_penalty = normalized_danger * wall_weight
                    
                    turn_penalty = 0
                    # 이전 이동 벡터(pdx, pdy)가 있을 경우, 현재 벡터(dx, dy)와의 내적을 통한 회전각 계산
                    if pdx != 0 or pdy != 0:
                        dot_product = (pdx*dx + pdy*dy) / (np.hypot(pdx, pdy) * np.hypot(dx, dy))
                        turn_penalty = (1 - dot_product) * turn_weight 
                    
                    # 경로 비용 합산
                    new_g_score = g_score + step_cost + turn_penalty + wall_penalty
                    next_state = (nx_x, nx_y, dx, dy)
                    
                    # 더 나은 경로(비용)일 경우 정보 갱신
                    if new_g_score < best_g_scores.get(next_state, float('inf')):
                        best_g_scores[next_state] = new_g_score
                        came_from[next_state] = state
                        
                        # 휴리스틱 (Euclidean distance to goal)
                        h_cost = np.hypot(goal[0]-nx_x, goal[1]-nx_y)
                        new_f_score = new_g_score + h_cost
                        
                        heapq.heappush(pq, (new_f_score, new_g_score, nx_x, nx_y, dx, dy))
                        
    # 큐가 빌 때까지 목표에 도달하지 못하면 경로 탐색 실패
    return float('inf'), []
