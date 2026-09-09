import cv2
import numpy as np

def decompose_to_convex(node, min_area_px, config):
    """
    주어진 노드의 driveable_mask가 볼록성이 낮은 경우, 재귀적으로 분할하여 볼록한 조각들로 나누는 함수임.
    - node: {'id': int, 'driveable_mask': np.ndarray, 'nondriveable_mask': np.ndarray, ...}
    - min_area_px: 분할된 조각이 너무 작아지는 것을 방지하기 위한 최소 면적 기준 (픽셀 단위)
    - 볼록성 계산은 contour와 convex hull을 이용하여 이루어지며, 일정 기준 이하인 경우에만 분할이 시도됨.
    - 예를 들어 solidity가 0.88보다 낮다는 것은, 실제 면적이 볼록 껍질 면적에 비해 88% 미만이라는 것을 의미함. => non-convexity
    - 분할은 수직 및 수평 투영을 분석하여 최적의 절단 축과 위치를 결정하며, L자 형태의 내부 코너도 고려하여 절단점을 선택함.
    - 반환값: 볼록한 조각들로 분할된 노드들의 리스트
    """
    mask = node['driveable_mask']
    non_drive = node.get(
        'nondriveable_mask',
        np.zeros_like(mask)
    )
    node_id = node.get('id', 'unknown')
    
    BASE_SOLIDITY = config.get('base_solidity_threshold', 0.88)
    STRICT_SOLIDITY = config.get('strict_solidity_threshold', 0.75)
    L_RATIO = config.get('l_shape_ratio_threshold', 0.85)
    PROJ_MARGIN = config.get('projection_margin_ratio', 0.1)
    BOTTLE_TOL = config.get('bottleneck_tolerance_ratio', 1.05)
    DIL_KERNEL_SIZE = config.get('dilation_kernel_size', 5)
    DIL_PADDING = config.get('dilation_padding_px', 10)
    
    # 1. 외곽선 검출 및 실제 Solidity 계산
    cnt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnt: 
        return [] # Contour가 없는 빈 마스크 폐기
    
    c = max(cnt, key=cv2.contourArea)
    area = cv2.contourArea(c)
    
    # 너무 작으면(min_area_px 미만인 경우) 최종 결과에서 제거. 분할 중단.
    if area < min_area_px: 
        return []
    
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 1.0
    
    # debug
    # print(f"DEBUG [Node ID: {node.get('id')}]:")
    # print(f" - Solidity: {solidity:.3f}")
    
    # debug
    original_nd_count = cv2.countNonZero(non_drive) # non-driveable_mask의 원래 핑크 픽셀 수
    # 자꾸 유실되어서 디버깅함.

    # 2. 볼록성이 낮을 경우 분할
    if solidity < BASE_SOLIDITY:
        x, y, w, h = cv2.boundingRect(c)
        
        # debug
        # print(f" - W/H Ratio: {w/h:.2f}")
        roi = mask[y:y+h, x:x+w]
        
        # 양방향 검사 및 투영 데이터 추출
        # 수직 수평 두 방향 모두의 투영(Projection)을 판단하여 최적의 절단 축과 위치 결정
        proj_h = np.sum(roi > 0, axis=0) # 세로 방향 절단을 위한 가로축 투영
        proj_v = np.sum(roi > 0, axis=1) # 가로 방향 절단을 위한 세로축 투영

        def analyze_projection(proj):
            """투영 배열 분석: 최적의 절단점과 비율을 반환함"""
            s = int(len(proj) * PROJ_MARGIN)
            e = int(len(proj) * (1 - PROJ_MARGIN))
            if e <= s: return 1.0, 0, 0, s
            
            search_area = proj[s:e]
            min_val = np.min(search_area)
            avg_val = np.mean(proj)
            ratio = min_val / avg_val if avg_val > 0 else 1.0
            
            # 최솟값과 거의 동일한 값들이 연속적으로 존재하는 구간의 경계를 탐색.
            # 이는 L자 형태의 내부 코너에서 나타날 수 있음.
            min_indices = np.where(search_area <= min_val * BOTTLE_TOL)[0]
            
            if len(min_indices) > 0:
                # 1. 최솟값 구간이 왼쪽 탐색 한계선(10%)에 닿아있는 경우 -> L자의 한쪽 팔
                if min_indices[0] == 0 and min_indices[-1] < len(search_area) - 1:
                    best_idx = min_indices[-1] # 구간이 끝나는 지점(내부 코너)에서 절단
                # 2. 최솟값 구간이 오른쪽 탐색 한계선(90%)에 닿아있는 경우 -> 반대쪽 팔
                elif min_indices[-1] == len(search_area) - 1 and min_indices[0] > 0:
                    best_idx = min_indices[0]  # 구간이 시작하는 지점(내부 코너)에서 절단
                # 3. 가운데에 뚜렷한 병목이 있거나 전체가 100% 평평한 경우
                else:
                    best_idx = min_indices[len(min_indices) // 2]
            else:
                best_idx = np.argmin(search_area)
                
            return ratio, min_val, avg_val, s + best_idx

        # 수직 수평 두 방향 모두의 투영 분석 결과
        ratio_h, min_width_h, avg_width_h, split_idx_h = analyze_projection(proj_h)
        ratio_v, min_width_v, avg_width_v, split_idx_v = analyze_projection(proj_v)

        # 더 뚜렷한 bottleneck(비율이 작은 쪽)을 절단 축으로 선택
        if ratio_h <= ratio_v:
            axis_horizontal = True # 가로 투영 사용 = 세로선 긋기
            best_ratio = ratio_h
            min_width, avg_width = min_width_h, avg_width_h
            relative_mid = split_idx_h
        else:
            axis_horizontal = False # 세로 투영 사용 = 가로선 긋기
            best_ratio = ratio_v
            min_width, avg_width = min_width_v, avg_width_v
            relative_mid = split_idx_v

        # L자 처리
        if best_ratio < L_RATIO or solidity < STRICT_SOLIDITY:
            m1 = np.zeros_like(mask)
            m2 = np.zeros_like(mask)

            # 주행 영역 분할: n1, n2는 일단 전체 non_drive를 복사하여 유실 방지
            if axis_horizontal:
                # 가로로 기니까 세로(Vertical)로 절단
                split_idx = x + relative_mid
                m1[:, :split_idx] = mask[:, :split_idx]
                m2[:, split_idx:] = mask[:, split_idx:]
                # 벽면 데이터는 일단 전체를 들고 감 (이후 connectedComponents 단계에서 필터링)
                temp_n1, temp_n2 = non_drive.copy(), non_drive.copy()
            else:
                # 세로로 기니까 가로(Horizontal)로 절단
                split_idx = y + relative_mid
                m1[:split_idx, :] = mask[:split_idx, :]
                m2[split_idx:, :] = mask[split_idx:, :]
                temp_n1, temp_n2 = non_drive.copy(), non_drive.copy()

            # U자 복도에서, 가로로 긴 복도 부분을 먼저 분할해 잘라냈다고 했을 때, 나머지 부분에 대해 생각해보자.
            # 세로의 두 복도 A와 B로 분리되어 연결되지 않은 경우, 한 노드로 두면 안되기에, 두 노드A와 B로 분리해줘야 하는 일이 있을 수 있음.
            # 이처럼 절단 이후, 남아있는 나머지가 모두 한 덩어리로 연결되었는지 확인 후, 
            # 만약 연결이 끊긴 부분이 있다면 독립된 id를 부여해 완전히 분리하는 과정임.
            
            # 이때, A와 B의 벽 모두 non-driveable_mask에 포함되어야 하는데, 단순히 절단선을 기준으로 나누면 벽이 한쪽에만 포함되는 문제가 발생할 수 있음.
            # 따라서, 절단선 주변의 일정 영역을 확장하여 비주행 마스크에도 포함시키는 방식으로, 절단된 두 노드 모두에 벽이 포함되도록 처리함.
            # 이렇게 하면, 절단된 노드들이 실제로는 연결되어 있더라도, 벽이 양쪽에 포함되어 있기 때문에, 후속 단계에서 연결이 끊긴 것으로 인식되어 독립된 노드로 분리될 수 있음.
            res_nodes = []
            for i, (m, n_all) in enumerate([(m1, temp_n1), (m2, temp_n2)]):
                if cv2.countNonZero(m) > 0:
                    num_labels, labels = cv2.connectedComponents(m, connectivity=8)
                    for j in range(1, num_labels):
                        sub_m = np.zeros_like(m)
                        sub_m[labels == j] = 255

                        sub_area = cv2.countNonZero(sub_m)
                        if sub_area < min_area_px:
                            print(f"  -> [PRE-DISCARD] Child {node_id}_{i}_{j} (Area: {sub_area} px) is smaller than min_area_px. Skipping.")
                            continue
                        
                        # 2. 유동적 Iteration 계산
                        # 노드 크기의 절반 정도는 팽창해야 벽에 닿을 수 있음
                        pts = cv2.findNonZero(sub_m)
                        if pts is None:
                            continue
                        _, _, sub_w, sub_h = cv2.boundingRect(pts)
                        dynamic_iter = max(sub_w, sub_h) // 2 + DIL_PADDING # DIL_PADDING: 최소 10px 여유
                        
                        # 3. 필터링 로직 수행
                        dil_kernel = np.ones((DIL_KERNEL_SIZE, DIL_KERNEL_SIZE), np.uint8)
                        expanded_sub_m = cv2.dilate(sub_m, dil_kernel, iterations=int(dynamic_iter))
                        sub_n = cv2.bitwise_and(n_all, expanded_sub_m)
                        
                        child_node = {
                            'id': f"{node_id}_{i}_{j}",
                            'driveable_mask': sub_m,
                            'nondriveable_mask': sub_n,
                            'area_px': cv2.countNonZero(sub_m)
                        }
                        res_nodes.extend(decompose_to_convex(child_node, min_area_px, config))
            
            # DEBUG: non-driveable  최종 유실 확인
            # 시각화를 확인하니 이제 유실 없이 잘 나옴.
            # 출력된 합계가 늘어나긴함. 
            # 중복되어 합계가 커질 수 있으나, 로봇 입장에서는 '어느 쪽 노드에서 보든 이 벽은 내 근처에 있다'고 인식하게 되므로 훨씬 안전함.
            final_child_nd_count = sum(cv2.countNonZero(child['nondriveable_mask']) for child in res_nodes)
            if final_child_nd_count == 0:
                print(f"[CRITICAL] All pink pixels lost in Node {node.get('id')}")
            else:
                print(f"DEBUG: Pink pixels preserved (Original: {original_nd_count} -> Final Sum: {final_child_nd_count})")
            
            return res_nodes
        
    return [node]