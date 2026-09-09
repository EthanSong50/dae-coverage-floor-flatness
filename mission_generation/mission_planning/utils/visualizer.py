import os
import cv2
import numpy as np
import colorsys

def _render_full_viz(nodes, path_segments, global_mask):
    """
    내부 헬퍼 함수: 노드와 경로 데이터를 바탕으로 시각화용 RGB 이미지를 생성함.
    """
    h, w = global_mask.shape[:2]
    viz_mask = np.zeros((h, w, 3), dtype=np.uint8)
    num_nodes = len(nodes)
    
    # 1. 배경 노드 색칠 (HLS 컬러 공간 활용)
    total_nd_mask = np.zeros((h, w), dtype=np.uint8)
    for i, node in enumerate(nodes):
        # 노드별 고유 색상 생성 (시각적 구분을 위해 Hue값 분산)
        start_hue = 0.75 
        end_hue = 0.0     
        hue = start_hue - (i / (num_nodes - 1)) * (start_hue - end_hue) if num_nodes > 1 else start_hue
        
        bg_rgb = colorsys.hls_to_rgb(hue, 0.77, 1.0)
        bg_color = np.array([c * 255 for c in bg_rgb], dtype=np.uint8)[::-1] # RGB to BGR
        
        viz_d = node['driveable_mask']
        viz_nd = node.get('nondriveable_mask')
        
        viz_mask[viz_d > 0] = bg_color
        if viz_nd is not None:
            total_nd_mask = cv2.bitwise_or(total_nd_mask, viz_nd)

    # 비구동 영역(장애물) 표시 - Magenta
    viz_mask[total_nd_mask > 0] = [255, 0, 255]

    # 2. 경로 렌더링 (Coverage: 하양/빨강점, Transit: 초록색)
    for idx, segment in enumerate(path_segments):
        seg_type = segment['type']
        path = segment['path']
        
        if not path or len(path) < 2:
            continue
            
        if seg_type == 'coverage':
            # 측정 경로는 굵은 흰색 선
            for k in range(len(path) - 1):
                cv2.line(viz_mask, path[k], path[k+1], (255, 255, 255), 2, cv2.LINE_AA)
            
        elif seg_type == 'transit':
            record_pcd = segment.get('record_pcd', False)
            if record_pcd:
                # 측정 중인 경유 구간 - coverage와 동일한 흰색으로 표시
                for k in range(len(path) - 1):
                    pt1 = tuple(map(int, path[k])); pt2 = tuple(map(int, path[k + 1]))
                    cv2.line(viz_mask, pt1, pt2, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(viz_mask, tuple(map(int, path[0])), 3, (0, 165, 255), -1)   # 시작: 주황 점
                cv2.circle(viz_mask, tuple(map(int, path[-1])), 3, (0, 165, 255), -1)  # 끝: 주황 점
            else:
                for k in range(len(path) - 1):
                    pt1 = tuple(map(int, path[k])); pt2 = tuple(map(int, path[k + 1]))
                    cv2.line(viz_mask, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

        elif seg_type == 'repass_preview':
            # boundary_repass(mission_execution/utils/boundary_repass.py)가 실행 시
            # 만들 왕복(러닝스타트/되짚기) 예상 경로 - 시안색 화살표로 별도 표시.
            # path[0]=coverage 시작/끝점, path[1]=예상 runway/retrace 지점.
            pt0 = tuple(map(int, path[0])); pt1 = tuple(map(int, path[1]))
            cv2.arrowedLine(viz_mask, pt0, pt1, (255, 255, 0), 2, cv2.LINE_AA, tipLength=0.15)
            cv2.circle(viz_mask, pt1, 4, (255, 255, 0), -1)
            label = segment.get('label', 'repass')
            cv2.putText(viz_mask, label, (pt1[0] + 6, pt1[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(viz_mask, label, (pt1[0] + 6, pt1[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    # 3. 노드 ID 표시
    for i, node in enumerate(nodes):
        mask = node['driveable_mask']
        if cv2.countNonZero(mask) > 0:
            M = cv2.moments(mask)
            if M["m00"] != 0:
                cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                text = f"ID:{node['id']}"
                cv2.putText(viz_mask, text, (cX - 20, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
                cv2.putText(viz_mask, text, (cX - 20, cY + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # 4. 미션 수행 순서(Sequence) 번호 매기기
    target_points = []
    for seg in path_segments:
        if seg['type'] == 'coverage' and len(seg['path']) > 0:
            target_points.append(seg['path'][0])
            if seg['path'][0] != seg['path'][-1]:
                target_points.append(seg['path'][-1])
    
    for idx, pt in enumerate(target_points):
        # 처음과 마지막 지점은 노란색, 중간은 빨간색
        is_edge = (idx == 0 or idx == len(target_points) - 1)
        color = (0, 255, 255) if is_edge else (0, 0, 255)
        
        cv2.circle(viz_mask, pt, 2 if is_edge else 1, color, -1)
        cv2.putText(viz_mask, str(idx + 1), (pt[0] - 8, pt[1] + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
        cv2.putText(viz_mask, str(idx + 1), (pt[0] - 8, pt[1] + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return viz_mask

def save_debug_image(nodes, path_segments, global_mask, output_dir="./debug_image", filename="full_mission_connected.png"):
    """
    최종 미션 상태를 이미지 파일로 저장함.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    viz_image = _render_full_viz(nodes, path_segments, global_mask)
    save_path = os.path.join(output_dir, filename)
    cv2.imwrite(save_path, viz_image)
    print(f"[*] Debug image saved to: {save_path}")
    return save_path

def plot_mission_state(nodes, path_segments, global_mask, wait_key=0):
    """
    현재 미션 상태를 화면에 표시함.
    """
    viz_image = _render_full_viz(nodes, path_segments, global_mask)
    cv2.imshow("Mission State Visualization", viz_image)
    cv2.waitKey(wait_key)
    return viz_image

def draw_waypoint_on_image(viz_image, pixel_points):
    """
    이미지 위에 샘플링된 경로 포인트들을 그림.
    """
    for pt in pixel_points:
        # (x, y) 좌표에 점을 찍음 (색상: 검정색)
        cv2.circle(viz_image, pt, 2, (0, 0, 0), -1)
    return viz_image