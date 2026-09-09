import numpy as np
import cv2
import math

def get_centroid(mask):
    """
    이진 마스크의 모멘트를 계산해 기하학적 중심점(Centroid)을 반환함.
    Coverage 알고리즘이 실패했을 때 해당 노드의 대표 지점으로 활용됨.
    """
    if mask is None or cv2.countNonZero(mask) == 0:
        return None
        
    M = cv2.moments(mask)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    return None

def euclidean_distance(p1, p2):
    """
    두 점 사이의 유클리디안 거리를 계산함. $d = \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}$
    """
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_distance_matrix(points):
    """
    점들의 리스트를 받아 TSP 알고리즘에 필요한 거리 행렬(Distance Matrix)을 생성함.
    """
    n = len(points)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = euclidean_distance(points[i], points[j])
            matrix[i][j] = matrix[j][i] = dist
    return matrix

def get_nearest_point(target_pt, points_list):
    """
    기준점(target_pt)에서 가장 가까운 점을 리스트(points_list)에서 찾아 반환함.
    Transit 경로의 시작점과 Coverage 경로의 끝점을 연결할 때 사용됨.
    """
    if not points_list:
        return None
    
    # 람다 식을 이용한 최소 거리 탐색
    nearest = min(points_list, key=lambda p: euclidean_distance(target_pt, p))
    return nearest

def pixel_to_meter(px_coord, origin, resolution, map_height):
    """
    이미지 픽셀 좌표를 실제 지도 상의 미터(m) 좌표로 정확히 역변환함(Y축 반전 보정).
    """
    mx = origin[0] + px_coord[0] * resolution
    # OpenCV Y축 인덱스를 ROS 2 물리 Y 좌표로 역산
    my = origin[1] + (map_height - 1 - px_coord[1]) * resolution
    return (mx, my)

def meter_to_pixel(m_coord, origin, resolution, map_height):
    """
    실제 미터(m) 좌표를 이미지 픽셀 좌표로 정확히 변환함(Y축 반전 반영).
    """
    px = int((m_coord[0] - origin[0]) / resolution)
    # ROS 2 물리 Y 좌표를 OpenCV Y축 인덱스로 변환
    py = int(map_height - 1 - ((m_coord[1] - origin[1]) / resolution))
    return (px, py)

def estimate_min_width_px(mask):
    """safe_node_mask(마진 적용 후)의 최소 폭을 minAreaRect로 근사."""
    if mask is None or cv2.countNonZero(mask) == 0:
        return 0.0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    return float(min(w, h))

def get_long_axis_angle_rad(mask):
    """
    mask를 감싸는 minAreaRect의 긴 변 방향을 라디안으로 반환함.
    narrow/ultra_narrow 노드에서 F2C 스와스 방향을 이 각도로 강제 지정하는 데 씀.

    주의: cv2.minAreaRect의 angle 규약이 OpenCV 버전마다 다름(4.5 이전과
    이후가 다름). 적용 전에 방향을 아는 노드 하나로 실제 결과를 시각화해서
    눈으로 확인할 것 - 90도 어긋나면 오히려 짧은 축 방향이 강제되어 역효과가 남.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    c = max(contours, key=cv2.contourArea)
    (_, _), (w, h), angle = cv2.minAreaRect(c)
    if w < h:
        angle += 90.0
    return math.radians(angle)

def order_swaths_by_entry(swath_pairs, entry_hint):
    """
    entry_hint에서 가장 가까운 스와스/끝점부터 출발해, 매번 남은 스와스 중
    현재 위치에서 가장 가까운 끝점을 갖는 스와스를 다음으로 고르는
    nearest-neighbor 체이닝. exit_hint는 고려하지 않음 - 진입 후 최대한
    빨리 커버리지를 시작하는 것이 목표이고, 노드 진출 방향은 신경 쓰지 않기로
    했기 때문임(왕복 방식으로 narrow/ultra_narrow의 blind zone을 커버하는
    전략과 일관됨).
    """
    n = len(swath_pairs)
    if n == 0:
        return []

    remaining = list(range(n))

    def pick_nearest(ref_pt):
        best_i, best_flip, best_d = None, False, float('inf')
        for idx in remaining:
            p1, p2 = swath_pairs[idx]
            d1 = euclidean_distance(ref_pt, p1)
            d2 = euclidean_distance(ref_pt, p2)
            if d1 < best_d:
                best_i, best_flip, best_d = idx, False, d1
            if d2 < best_d:
                best_i, best_flip, best_d = idx, True, d2
        return best_i, best_flip

    ordered = []
    ref = entry_hint if entry_hint is not None else swath_pairs[0][0]

    while remaining:
        idx, flip = pick_nearest(ref)
        p1, p2 = swath_pairs[idx]
        if flip:
            p1, p2 = p2, p1
        ordered.append((p1, p2))
        remaining.remove(idx)
        ref = p2

    return ordered