import cv2
import numpy as np

def cut_holes(node, resolution):
    """
    실내 공간의 구멍(Hole, 섬)을 탐지하고, 이 도넛 형태의 공간의 bounding box에서, 가장 가까운 외벽으로 절단선(Cut line)을 생성하여, 이를 기반으로 공간을 분할하는 함수.
    1. 구멍 탐지: 각 노드의 주행 가능 영역과 비주행 영역을 합쳐 '전체 방 덩어리'를 만들고, cv2.findContours와 RETR_CCOMP 모드를 사용하여 외곽선과 내부 구멍을 계층적으로 탐지함.
    2. 절단선 생성: 구멍의 외곽선 모든 픽셀에서 사방으로 레이저를 쏘아, 가장 짧은 거리를 가진 절단선만 채택하여, 도넛 공간을 분할함.
    3. 물리적 절단 및 독립 ID 부여: 마스크에 직접 검은색 선을 그려서 공간을 분리. 단순한 선이 아닌, 두께 2로 하여 확실하게 절단함. 그리고 Connected Components로 분리된 각각의 조각들을 새로운 노드로 등록함.
    4. Gap Filling: 절단선 영역이 0이 된 후, 아직 해당 영역은 어떤 노드에도 할당이 안 된 상태이므로, 절단선 영역의 픽셀들이 가장 가까운 노드에 할당되도록 함.
    - Iterative Growth 방식: 간단하게 반복적으로 주변 픽셀에서 값을 가져와서 할당하는 방식을 사용함. (왜냐하면 절단선이 2픽셀로 두껍기 때문에, 한 번의 팽창으로도 대부분의 픽셀이 할당될 수 있기 때문.)
    5. 최종 노드 반환.
    """
    new_nodes = []
    d_mask = node['driveable_mask'].copy()
    nd_mask = node['nondriveable_mask'].copy()
        
    # 주행 영역과 비주행 영역을 합친 '전체 방 덩어리' 생성 (구멍 탐지용)
    total_mask = cv2.bitwise_or(d_mask, nd_mask)
        
    # 1. 외곽선 및 내부 구멍(Hole) 계층 구조 탐색
    # RETR_CCOMP: 외곽 테두리와 내부 구멍 테두리를 2단계 계층으로 완벽히 분리. 
    # 계층적으로, 즉 부모와 자식 관계를 통해 실내 공간 안의 구멍을 식별함.
    contours, hierarchy = cv2.findContours(total_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
    cut_lines = []
    has_hole = False
        
    if hierarchy is not None:
        # hierarchy[0][i] = [Next, Previous, First_Child, Parent] # 계층 구조
        for i in range(len(contours)):
            parent_idx = hierarchy[0][i][3]
                
            # Parent가 존재하면(-1이 아니면) 해당 윤곽선은 도형 내부의 '구멍'임
            if parent_idx != -1:
                # 미세한 픽셀 노이즈는 섬으로 취급하지 않음 (예: 0.5 제곱미터 미만 무시)
                # 구멍의 면적 계산 (단위: 픽셀 -> 제곱미터)
                area = cv2.contourArea(contours[i])
                min_hole_px = 0.5 / (resolution * resolution)
                if area < min_hole_px:
                    continue
                        
                has_hole = True
                    
                # 2. 구멍(섬)의 Bounding Box 추출
                x, y, w, h = cv2.boundingRect(contours[i])
                
                # 3. Ray-casting
                # 닿는 지점(검은색 픽셀 = 벽)을 찾아 절단선으로 저장
                # 가장 가까운 외벽까지의 절단선을 생성하여, 도넛 공간을 분할함.
                # 물론 실제로는 도넛이 아닌, non-convex한 공간 안에 non-convex한 구멍이 있는 경우라고 가정.
                # 구멍의 외곽선 모든 픽셀에서 사방으로 레이저를 쏘아, 가장 짧은 거리를 가진 절단선만 채택
                best_cuts = {'up': None, 'down': None, 'left': None, 'right': None}
                min_dists = {'up': float('inf'), 'down': float('inf'), 'left': float('inf'), 'right': float('inf')}
                
                hole_contour = contours[i]
                
                for pt in hole_contour:
                    px, py = pt[0]
                    
                    # 상(Up) 탐색: 바로 위 픽셀이 주행 공간(>0)일 때만 밖으로 발사
                    if py > 0 and total_mask[py - 1, px] != 0:
                        for ty in range(py - 1, -1, -1):
                            if total_mask[ty, px] == 0:
                                dist = py - ty
                                if dist < min_dists['up']:
                                    min_dists['up'] = dist
                                    best_cuts['up'] = ((px, py), (px, ty))
                                break
                    
                    # 하(Down) 탐색
                    if py < total_mask.shape[0] - 1 and total_mask[py + 1, px] != 0:
                        for ty in range(py + 1, total_mask.shape[0]):
                            if total_mask[ty, px] == 0:
                                dist = ty - py
                                if dist < min_dists['down']:
                                    min_dists['down'] = dist
                                    best_cuts['down'] = ((px, py), (px, ty))
                                break
                    
                    # 좌(Left) 탐색
                    if px > 0 and total_mask[py, px - 1] != 0:
                        for tx in range(px - 1, -1, -1):
                            if total_mask[py, tx] == 0:
                                dist = px - tx
                                if dist < min_dists['left']:
                                    min_dists['left'] = dist
                                    best_cuts['left'] = ((px, py), (tx, py))
                                break
                    
                    # 우(Right) 탐색
                    if px < total_mask.shape[1] - 1 and total_mask[py, px + 1] != 0:
                        for tx in range(px + 1, total_mask.shape[1]):
                            if total_mask[py, tx] == 0:
                                dist = tx - px
                                if dist < min_dists['right']:
                                    min_dists['right'] = dist
                                    best_cuts['right'] = ((px, py), (tx, py))
                                break
                
                # 4방향 중 유효하게 찾아낸 최단 거리 절단선만 배열에 추가
                for direction, cut_line in best_cuts.items():
                    if cut_line is not None:
                        cut_lines.append(cut_line)

    # 4. 물리적 절단 및 독립 ID 부여
    if has_hole and cut_lines:
        print(f"    -> Found hole(s). Applying {len(cut_lines)} cut lines.")
        
        # 절단 전의 원본 마스크
        original_d_mask = node['driveable_mask'].copy()
        original_nd_mask = node['nondriveable_mask'].copy()
        
        # 마스크에 직접 검은색 선을 그려서 공간을 분리.
        # 단순한 선이 아닌, 두께 2로 하여 확실하게 절단함. 
        # 왜냐하면 해상도 이슈 떄문에 1픽셀로는 완벽하게 절단되지 않을 수 있기 때문임.
        cut_total_mask = total_mask.copy()
        for start_pt, end_pt in cut_lines:
            cv2.line(cut_total_mask, start_pt, end_pt, 0, thickness=2)
                
        # 오직 절단선 영역만을 남긴 마스크 (= 절단 후 0이 된 영역)
        cut_mask = cv2.bitwise_and(total_mask, cv2.bitwise_not(cut_total_mask))
        
        # 연결성 8(connectivity=8)을 사용하여 대각선 고립 방지
        num_labels, labels = cv2.connectedComponents(cut_total_mask, connectivity=8)
        
        # Gap Filling 로직
        # labels에서 0인 영역 중 total_mask가 1인 곳(절단선 영역)을 가장 가까운 label로 채움
        refined_labels = labels.copy().astype(np.int32)
            
        # 주행 가능 영역인데 현재 label이 0(절단선/벽)인 픽셀들 좌표 추출
        unassigned_mask = (total_mask > 0) & (labels == 0)
            
        if np.any(unassigned_mask):
            # Distance Transform을 사용하여 가장 가까운 label을 찾는 방식.
            # labels가 0인 곳으로부터 가장 가까운 'labels > 0'인 곳의 인덱스를 반환.
            dist, labels_map = cv2.distanceTransformWithLabels(
                (labels == 0).astype(np.uint8), 
                cv2.DIST_L2, 
                5, 
                labelType=cv2.DIST_LABEL_CCOMP
            )
            # labels_map은 각 픽셀에서 가장 가까운 0-픽셀의 '연결된 성분 ID'를 줌.
            # 이를 이용해 모든 픽셀에 대해 전역적인 할당을 수행하는 방식은 복잡하므로 사용하지 않고,
            # Iterative Growth 방식을 대신 사용하여 모든 픽셀을 각 노드에 할당함.
            for _ in range(3):  # 두께가 2px이므로 2~3회 정도.
                curr_unassigned = (total_mask > 0) & (refined_labels == 0)
                if not np.any(curr_unassigned): break
                    
                # 주변 픽셀(상하좌우)에서 값을 가져옴
                expanded = cv2.dilate(refined_labels.astype(np.float32), np.ones((3,3)))
                refined_labels[curr_unassigned] = expanded[curr_unassigned].astype(np.int32)
            
        # 분리된 각각의 조각들을 새로운 노드로 등록.
        for j in range(1, num_labels):
            # 이제 refined_labels를 사용하여 틈새 없이 영역을 가져옴
            final_node_area = (refined_labels == j).astype(np.uint8) * 255
            
            # '절단선이 없는 깨끗한 원본'에서 해당 영역의 픽셀만 추출
            sub_d = cv2.bitwise_and(original_d_mask, final_node_area)
            sub_nd = cv2.bitwise_and(original_nd_mask, final_node_area)
            
            if cv2.countNonZero(sub_d) > 0:
                new_nodes.append({
                    'id': None, # ID는 나중에 space_segmenter에서 부여됨
                    'driveable_mask': sub_d,
                    'nondriveable_mask': sub_nd,
                    'area_px': cv2.countNonZero(sub_d)
                })
        return new_nodes
    else:
        # 내부에 구멍이 없는 단순 다각형(Simple Polygon) 공간은 기존 id체계로 재등록시킴.
        return [node]
