# mission_generation/mission_planning/translator.py

import math
from mission_planning.utils import geometry

def calculate_yaw(p1_m, p2_m):
    dx = p2_m[0] - p1_m[0]
    dy = p2_m[1] - p1_m[1]
    return math.atan2(dy, dx)

def euler_to_quaternion(yaw):
    return {
        'x': 0.0, 'y': 0.0,
        'z': math.sin(yaw / 2.0),
        'w': math.cos(yaw / 2.0)
    }

def convert_segments_to_nav2(path_segments, origin, resolution, map_height):
    """
    전체를 flatten하지 않고 세그먼트 본연의 데이터 구조를 유지한 상태로 메트릭(Meter)
    좌표 및 Quaternion 보정을 수행함.
    """
    translated_segments = []
    
    for segment in path_segments:
        seg_type = segment['type']
        pixel_path = segment['path']
        record_pcd = segment.get('record_pcd', seg_type == 'coverage')
        if not pixel_path:
            continue
            
        segment_poses = []
        for i in range(len(pixel_path)):
            px_coord = pixel_path[i]
            m_coord = geometry.pixel_to_meter(px_coord, origin, resolution, map_height)        
            
            # 방향(Orientation) 계산
            if i < len(pixel_path) - 1:
                next_m_coord = geometry.pixel_to_meter(pixel_path[i+1], origin, resolution, map_height)
                yaw = calculate_yaw(m_coord, next_m_coord)
            else:
                if i > 0:
                    prev_m_coord = geometry.pixel_to_meter(pixel_path[i-1], origin, resolution, map_height)
                    yaw = calculate_yaw(prev_m_coord, m_coord)
                else:
                    yaw = 0.0
                    
            quaternion = euler_to_quaternion(yaw)
            
            pose_dict = {
                'pose': {
                    'position': {'x': m_coord[0], 'y': m_coord[1], 'z': 0.0},
                    'orientation': quaternion
                }
            }
            segment_poses.append(pose_dict)
            
        # 변환된 데이터를 세그먼트 래퍼에 감싸서 보존
        # node_id(coverage)/from_node_id,to_node_id(transit)는 실행 시점
        # 디버그 로그(mission_execution/utils/mission_logger.py)가 "지금 몇 번
        # 노드를 처리 중인지"를 표시하는 데 씀 - 있으면 그대로 전달, 없으면 None.
        translated_segments.append({
            'type': seg_type,
            'poses': segment_poses,
            'record_pcd': record_pcd,
            'node_id': segment.get('node_id'),
            'from_node_id': segment.get('from_node_id'),
            'to_node_id': segment.get('to_node_id'),
        })

    return translated_segments
