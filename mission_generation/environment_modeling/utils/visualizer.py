import cv2
import numpy as np
import colorsys
import os
# 시각화 유틸리티 함수들.
# 맵 분할 파이프라인의 각 단계에서 중간 결과를 시각화해 디버깅 및 분석에 활용하기 위한 함수들임.
# visualize_geometry_view: 단일 마스크나 이미지를 저장하는 함수. 맵 분할 파이프라인의 Step 1과 Step 2에서 사용됨 - 주행 가능 영역과 비주행 영역을 구분해 시각화할 때 활용됨.
# visualize_node_view: 여러 노드를 시각화하는 함수. 맵 분할 파이프라인의 Step 3부터 Step 6까지 사용됨.
def visualize_geometry_view(image, step_name, save_path):
    """
    단순 마스크나 이미지를 지정된 경로에 저장함. (맵 분할 파이프라인 Step 1, 2용)
    """
    if image is None:
        return
        
    filename = f"step_{step_name}.png"
    full_path = os.path.join(save_path, filename)
    
    # 1채널 마스크인 경우도 cv2가 Grayscale로 저장.
    cv2.imwrite(full_path, image)
    print(f"    -> [Debug Save] {filename}")

def visualize_node_view(nodes, map_shape, step_name, save_path):
    """
    모든 노드를 시각화하고 이미지로 저장. ()맵 분할 파이프라인 Step 3-6용)
    """
    h, w = map_shape
    viz_mask = np.zeros((h, w, 3), dtype=np.uint8)
    num_nodes = len(nodes)

    # [DEBUG] 전체 비주행 영역 통합 마스크 생성
    total_nd_mask = np.zeros((h, w), dtype=np.uint8)

    for i, node in enumerate(nodes):
        # 1. 색상 결정
        start_hue = 0.75  # 보라색 부터
        end_hue = 0.0     # 빨강색까지.
        if num_nodes > 1:
            hue = start_hue - (i / (num_nodes - 1)) * (start_hue - end_hue)
        else:
            hue = start_hue # 노드가 하나면 그냥 색상 고정
            
        rgb = colorsys.hls_to_rgb(hue, 0.77, 1.0) # 명도와 채도값 조정
        color_arr = np.array([c * 255 for c in rgb], dtype=np.float32)

        # 2. 영역 칠하기
        viz_d = node['driveable_mask']
        viz_nd = node['nondriveable_mask']
        # 주행 가능 영역
        viz_mask[viz_d > 0] = color_arr[::-1] 
        # 비주행 영역: 핑크
        total_nd_mask = cv2.bitwise_or(total_nd_mask, viz_nd) # 누적.
        
    # 모든 노드를 다 순회한 '마지막'에 핑크색을 입혀서 덮어쓰기 방지
    viz_mask[total_nd_mask > 0] = [255, 0, 255]

    # 3. ID 텍스트 삽입 (가독성을 위해 별도 루프)
    for i, node in enumerate(nodes):
        mask = node['driveable_mask']
        if cv2.countNonZero(mask) > 0:
            M = cv2.moments(mask) # 무게 중심 계산
            if M["m00"] != 0:
                cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                text = f"ID:{node.get('id', i+1)}"
                # 검은색 외곽선이 있는 흰색 글자
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8  # 폰트 크기 조절
                thickness = 2  # 글자 두께
                
                (t_w, t_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_pos = (cX - t_w // 2, cY + t_h // 2)
                
                # 텍스트 외곽선(검은색) 및 본문(흰색)
                cv2.putText(viz_mask, text, text_pos, font, font_scale, (0, 0, 0), thickness + 4)
                cv2.putText(viz_mask, text, text_pos, font, font_scale, (255, 255, 255), thickness)
            

    # 4. 저장
    full_path = os.path.join(save_path, f"step_{step_name}.png")
    cv2.imwrite(full_path, viz_mask)
    return viz_mask
