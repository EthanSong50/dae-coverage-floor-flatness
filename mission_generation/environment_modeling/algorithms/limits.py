import cv2
import numpy as np

def enforce_physical_limits(mask, robot_px):
    """
    이전 단계에서는 벽과 기둥 안의 공간을 제외시키기 위해 단순히 가장 큰 연결된 실내 바닥 영역을 추출했지만, 
    실제 로봇이 주행 가능한 영역과 그렇지 않은 영역을 구분함(둘 다 최대한 측정 대상임).
    예를 들어, 문지방이 로봇의 폭에 비해 너무 좁아 진입이 불가하다면, 그 방 전체가 주행 불가 영역으로 간주될 수 있음.
    이때 이 방과 방 바깥 중에서 더 넓은 영역(복도, 거실 등)이 주행 가능 영역으로 선택됨.
    """
    print("\n[Step 1] Enforcing Physical Limits...")
    
    # 1. 거리 변환 (벽면으로부터의 거리)
    robot_radius_px = (robot_px / 2.0) # 로봇 반경
    robot_radius_px = robot_radius_px * 1.1 # 안전 마진 곱해줌.
    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    
    # 2. 로봇 반경 이상 확보된 주행 가능 영역 추출
    safe_mask = np.zeros_like(mask)
    safe_mask[dist_map > robot_radius_px] = 255
    
    # 3. 연결성 검사: 가장 큰 주행 영역만 유지
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(safe_mask, connectivity=8)
    
    driveable_mask = np.zeros_like(mask)
    if num_labels > 1:
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        driveable_mask[labels == largest_label] = 255
    else:
        print(" [!] Warning: No driveable area found!")

    # 4. nondriveable 영역 계산 = (원본 바닥) - (주행 영역)
    nondriveable_mask = cv2.bitwise_and(mask, cv2.bitwise_not(driveable_mask))
    # 여기서 nondriveable_mask 영역은 로봇이 주행이 불가하지만, 그래도 센싱을 최대한 해야 하는 영역을 의미 (벽 근처 등등)

    # 5. 디버그용 컬러 이미지 생성 - 배경은 검정.
    debug_color = cv2.cvtColor(np.zeros_like(mask), cv2.COLOR_GRAY2BGR)
    debug_color[driveable_mask == 255] = [255, 255, 255] # 주행 가능: 흰색
    
    # 주행 불가 지역: 핑크색 (BGR)
    if cv2.countNonZero(nondriveable_mask) > 0:
        debug_color[nondriveable_mask == 255] = [255, 0, 255]
    else:
        print(" [!] Note: No nondriveable area detected, check the input mask or parameters.")
        
    print(f"    -> [Data Check] robot_radius_px: {robot_radius_px} px")
    print(f"    -> [Final Check] Nondriveable Area: {cv2.countNonZero(nondriveable_mask)} px")
    
    return driveable_mask, nondriveable_mask, debug_color
