import os
import numpy as np
import cv2

# 분리된 알고리즘 모듈 임포트
from .utils import visualize_geometry_view, visualize_node_view
from .algorithms import (
    enforce_physical_limits,
    split_by_doors, 
    cut_holes, 
    decompose_to_convex, 
    subdivide_long_nodes
)

class SpaceSegmenter:
    def __init__(self, mask, resolution, origin, visualization_dir, config):
        # MapPreprocessor로부터 정제된 데이터를 받아 초기화.
        self.mask = mask
        self.resolution = resolution
        self.origin = origin
        self.visualization_dir = visualization_dir
        
        self.robot_width = config.get('robot_width', 0.2) # (key, default)
        self.max_door_radius = config.get('max_door_radius', 0.6) # 일반적인 방의 문 폭의 최대 너비의 절반: 1.2 / 2 = 0.6 (m)
        self.min_area_px = config.get('min_area_px', 100) # 분할된 조각이 너무 작아지는 것 방지.
        self.max_aspect_ratio = config.get('max_aspect_ratio', 5) # 공간의 최대 종횡비의 기준. 예를 들어 공간의 종횡비가 5 이상이라면, 지나치게 긴 공간으로 간주하여 분할함.

        self.convexity_config = config.get('convexity', {})
        
        # 픽셀 단위 변환
        calc_robot_px = self.robot_width / self.resolution
        self.robot_px = max(2, int(round(calc_robot_px)))
        
        print(f"[*] Space Segmentation Initialized: Resolution {self.resolution}m/px")
        print(f"[*] Debug images will be saved in: {self.visualization_dir}")
        
    def _assign_global_ids(self, nodes):
        """
        모든 노드에 대해 단순 정수형 고유 ID를 순차적으로 부여하도록 관리하는 함수.
        """
        for i, node in enumerate(nodes):
            node['id'] = i + 1  # 1-based index
        return nodes
        
    def execute_segmentation(self):
        """
        맵 분할 파이프라인 구성:
        Step 1: "로봇이 어디를 갈 수 있는가?"를 결정하는 단계(물리적 한계 적용).
        => .utils의 visualize_geometry_view으로 시각화

        Step 2-5: "공간을 어떻게 분할할 것인가?"를 결정하는 단계(문 기반 분할 + 구멍 탐지 및
        절단 + 볼록성 기반 분할 + 너무 긴 복도 분할).
        => .utils의 visualize_node_view으로 시각화

        각 Step의 상세 설명은 environment_modeler.py의 build_environment_model()
        참고(동일한 파이프라인 설명이 그쪽에도 있음).
        """
        print(f"\n{'-'*16} [Space Segmentation Start] {'-'*16}")

        # [Step 1] Driveable vs nondriveable 영역 분리
        # algorithms/limits.py 호출
        """
        이전 단계인 맵 전처리 단계(Pre-processing)에서는 벽과 기둥 안의 공간을 제외시키기 위해
        단순히 가장 큰 이어지는 실내 바닥 영역을 추출함.
        이번 단계에서는 여기서 실제 로봇이 주행 가능한 영역과 그렇지 않은 영역으로 구분함(둘 다
        최대한 측정 대상임).
        문지방이 로봇의 폭에 비해 너무 좁아 진입이 불가하면 그 방 전체가 주행 불가 영역으로
        간주될 수 있음 - 이때 이 방과 방 바깥 중에서 더 넓은 영역(복도, 거실 등)이 주행 가능
        영역으로 선택됨.
        """
        driveable_mask, nondriveable_mask, debug_color = enforce_physical_limits(
            self.mask, self.robot_px
        )
        visualize_geometry_view(debug_color, "1_physical_limits", self.visualization_dir) # 시각화 - .utils의 visualize_geometry_view 활용
        
        # [Step 2] 문(Door) 기반 1차 분할 및 노드 생성
        # algorithms/doors.py 호출
        """
        타겟을 방으로 삼아 공간을 분할함.
        방을 나누는 가장 명확한 기준은 문이므로, 문을 기준으로 공간을 분할해 노드를 생성함.
        문이 로봇이 통과할 수 있는지 여부를 판단해, 통과가 불가능한 문이 존재하는 방은 주행
        불가능 영역으로 간주함 - 이때 이 타겟과 전체 영역에서 타겟을 뺀 나머지 중 더 넓은
        영역(복도, 거실 등)이 주행 가능 영역으로 선택됨.
        """
        print("\n[Step 2] Splitting by doors and Geodesic sensing assignment...")
        nodes_after_doors = split_by_doors(
            driveable_mask,
            nondriveable_mask, 
            self.resolution, 
            self.max_door_radius, 
            segmenter=self
        )
        nodes_after_doors = self._assign_global_ids(nodes_after_doors) # 노드에 고유 ID 부여
        print(f"    -> Nodes after door checking and splitting: {len(nodes_after_doors)}")
        visualize_node_view(nodes_after_doors, self.mask.shape, "2_split_by_doors", self.visualization_dir) # 시각화
        
        # [Step 3] hole(구멍) 탐지 및 분할
        # algorithms/holes.py 호출
        """
        타겟은 실내 구멍이 있는 도넛형 공간임.
        이전 단계(nodes_after_doors)에서 생성된 노드들을 순회하며 구멍이 있으면 절단함.
        구멍 표면에서 가장 가까운 외벽을 찾아 절단선을 그은 후,
        Connected Components로 분할된 각각의 조각들을 새로운 노드로 등록하는 방식임.
        """
        print("\n[Step 3] Detecting and cutting holes (non-convex sections)...")
        nodes_after_holes = []
        for node in nodes_after_doors:
            # cut_holes가 단일 노드를 받아 분할된 리스트를 반환.
            result = cut_holes(node, self.resolution)
            nodes_after_holes.extend(result)
        nodes_after_holes = self._assign_global_ids(nodes_after_holes) # 노드에 고유 ID 부여
        print(f"    -> Nodes after hole checking and splitting: {len(nodes_after_holes)}")
        visualize_node_view(nodes_after_holes, self.mask.shape, "3_hole_splitting", self.visualization_dir) # 시각화
        
        
        # [Step 4] 볼록성(Solidity) 기반 오목한 공간(L자 복도 등) 분할
        """
        타겟은 L자형 복도 등 볼록성이 낮은 오목한 공간임.
        재귀적으로 볼록성이 낮은 노드에 대해 절단선을 그어 분할함.
        절단선은 구멍 탐지에서 사용한 방식과 유사하게, 볼록성이 낮은 영역의 표면에서 가장
        가까운 외벽을 찾아 그은 후, Connected Components로 분할된 각각의 조각들을 새로운
        노드로 등록하는 방식임.
        """
        # algorithms/convexity.py 호출
        print("\n[Step 4] Recursive decomposition for non-convex sections...")
        nodes_after_convexity = []
        for node in nodes_after_holes:        
            # 한 노드가 재귀적으로 여러 조각(List)으로 반환됨.
            pieces = decompose_to_convex(node, self.min_area_px, config = self.convexity_config)
            nodes_after_convexity.extend(pieces)
        nodes_after_convexity = self._assign_global_ids(nodes_after_convexity) # 노드에 고유 ID 부여
        print(f"    -> Nodes after recursive convexity checking and splitting: {len(nodes_after_convexity)}")
        visualize_node_view(nodes_after_convexity, self.mask.shape, "4_convexity_split", self.visualization_dir) # 시각화
        
        # [Step 5] 너무 긴 복도 분할
        # algorithms/subdivider.py 호출
        """
        타겟은 매우 긴 복도임.
        매우 긴 복도는 로봇 운용에 비효율적일 수 있으므로, 설정된 비율을 초과하는 노드를 분할함.
        """
        print("\n[Step 5] Subdividing long/linear nodes for operational efficiency...")
        
        nodes_after_subdivision = subdivide_long_nodes(
            nodes_after_convexity, 
            max_aspect_ratio=self.max_aspect_ratio,
            min_area_px=self.min_area_px
        )
        nodes_after_subdivision = self._assign_global_ids(nodes_after_subdivision) # 노드에 고유 ID 부여
        print(f"    -> Nodes after long node subdivision: {len(nodes_after_subdivision)}")
        visualize_node_view(nodes_after_subdivision, self.mask.shape, "5_long_node_subdivision", self.visualization_dir) # 시각화
        
        # 파이프라인 종료. 최종 노드 반환
        final_nodes = nodes_after_subdivision
        print("\n[*] Space Segmentation executed successfully. Final nodes ready for topology saving.")
        return final_nodes
        
    def save_topology(self, nodes, output_dir=None):
        # 분할된 모든 노드를 .npz 파일로 저장.
        output_filename = 'final_topological_map.npz'
        # main.py가 있는 폴더 혹은 별도 지정 폴더에 저장
        full_output_path = os.path.join(output_dir, output_filename)
        
        driveable_masks = [n['driveable_mask'] for n in nodes]
        nondriveable_mask = [n['nondriveable_mask'] for n in nodes]
        
        np.savez(full_output_path, 
                 nodes=driveable_masks, 
                 nondriveable_nodes=nondriveable_mask, 
                 resolution=self.resolution, 
                 origin=self.origin)
        
        print(f"\n{'='*60}")
        print(f"[*] Success: {len(nodes)} nodes created.")
        print(f"[*] Final Map Saved: {full_output_path}")
        print(f"{'='*60}\n")
