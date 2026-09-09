# mission_generation/environment_modeling/environment_modeler.py

import os
import numpy as np
import traceback
from .map_preprocessor import MapPreprocessor
from .space_segmenter import SpaceSegmenter
from .map_generator import MapGenerator


class EnvironmentModeler:
    def __init__(self, config):
        # Global Workspace 설정 및 경로 로드 (~/dae_floor_maps)
        self.workspace_root = os.path.expanduser(config['global']['workspace_root'])

        self.cfg = config['environment_modeling']
        self.target_resolution = self.cfg['target_resolution']
        self.dae_file = self.cfg['dae_file']
        
        self.assets_dir = os.path.join(self.workspace_root, self.cfg['input_dir'])
        self.grid_dir = os.path.join(self.workspace_root, self.cfg['output_grid_dir'])
        self.topology_dir = os.path.join(self.workspace_root, self.cfg['output_topology_dir'])
        self.visualization_dir = os.path.join(self.workspace_root, self.cfg['visualization_dir'])

    def build_environment_model(self) -> bool:
        dae_path = os.path.join(self.assets_dir, self.dae_file)
        yaml_path = os.path.join(self.grid_dir, "map_from_dae.yaml") # 3D(.dae)에서 2D로 변환된 맵(pgm, yaml)

        print(f"[*] Using external workspace: {self.workspace_root}")
        print(f"[*] Target .dae file source: {dae_path}")
        print(f"[*] Starting Environment modeler... (Resolution: {self.target_resolution}m/px)")

        # --- [Step 1] 2D Map Generation from 3D Model(.dae file) ---
        # .dae 파일이 있으면 매번 2D 맵을 처음부터 재생성함 - 'and not
        # os.path.exists(yaml_path)' 조건은 의도적으로 주석 처리되어 있음(파라미터
        # 하나만 바꿔 재실행해도 노드 개수/연결 구조가 미세하게 달라질 수 있으니
        # 결과가 크게 다르면 이 재생성 자체를 먼저 의심할 것, CLAUDE.md 참고).
        print(f"--- [Step 1] 2D Map Generation from 3D Model ---")
        if os.path.exists(dae_path): # and not os.path.exists(yaml_path):
            try:
                generator = MapGenerator(
                    dae_path=dae_path, 
                    output_dir=self.grid_dir,
                    resolution=self.target_resolution
                    )
                yaml_path = generator.generate_2d_map()
            except Exception as e:
                print(f"[!] Map Generation Error: {e}")
                traceback.print_exc()
                return False
        elif not os.path.exists(yaml_path):
            # 수동 배치 유저를 위한 명확한 예외 안내 가이드라인 출력
            print(f"\n[!] Critical Error: DAE file or YAML file not found.")
            print(f"     - DAE file path: {dae_path}")
            print(f"     - YAML file path: {yaml_path}")
            return False

        # --- [Step 2] Map Pre-processing ---
        # 벽의 바깥쪽과 기둥 안의 공간 등 아예 측정 대상이 아닌 영역을 제거.
        # 이때 단순히 가장 큰 연결된 바닥 영역을 찾아, 이 부분을 측정 대상 영역으로 간주함.
        # 그리고 이 영역의 마스크를 반환하는 방식으로 진행함.
        print("--- [Step 2] Map Pre-processing ---")
        npz_path = os.path.join(self.grid_dir, "map_preprocessed_data.npz")
        
        # 이미 전처리된 npz 파일이 있다면 로드하여 연산 스킵 (Step3로 바로 넘어감)
        if os.path.exists(npz_path):
            print(f"[*] Found existing preprocessed data. Loading from: {npz_path}")
            data = np.load(npz_path)
            raw_mask = data['mask']
            # origin과 resolution은 preprocessor 클래스를 거치지 않았으므로 npz에서 직접 추출
            map_resolution = float(data['resolution'])
            map_origin = data['origin']
            
        else:
            print(f"[*] Target Map: {yaml_path}")
            try:
                preprocessor = MapPreprocessor(yaml_path)
                raw_mask, stats = preprocessor.get_largest_connected_area()
                
                if raw_mask is None:
                    print("[!] Critical Error: Failed to extract floor area.")
                    return False

                preprocessor.save_analysis_result(
                    mask=raw_mask, 
                    stats=stats, 
                    visualization_dir=self.visualization_dir,
                    data_dir=self.grid_dir
                )
                map_resolution = preprocessor.resolution
                map_origin = preprocessor.origin
                
            except Exception as e:
                print(f"[!] Map Preprocessing Error: {e}")
                traceback.print_exc()
                return False

        # --- [Step 3] Space Segmentation Process ---
        """
        (벽 바깥쪽, 벽이나 기둥 안의 공간을 제외한) 모든 실내 공간은 이전 단계에서 언급했듯이 측정 대상임.

        맵 분할 파이프라인의 구성:
        - Step 1: "로봇이 어디를 갈 수 있는가?"를 결정하는 단계(물리적 한계 적용).
            이 단계가 맵 분할 파이프라인이 아니라 맵 전처리 단계에 속하는 게 더 자연스러워 보일
            수 있으나, 맵 전처리는 지도 중심이고 이 단계는 로봇 중심이라는 차이가 있음.
        - Step 2~5: "공간을 어떻게 분할할 것인가?"를 결정하는 단계(문 기반 분할 + 구멍 탐지 및
            절단 + 볼록성 기반 분할 + 너무 긴 복도 분할).

        Step 1: 실내 공간을 주행 가능 영역과 주행이 불가능한 영역으로 나눔(둘 다 최대한 측정 대상임).
        Step 2: 문을 기준으로 공간을 분할해 노드를 생성함. 이때 문이 로봇이 통과할 수 있는지
            여부를 판단해, 통과가 불가능한 문이 존재하는 방은 주행 불가능 영역으로 간주될 수 있음.
        Step 3: hole(구멍) 탐지 및 분할. 이전 단계에서 생성된 노드들을 순회하며 구멍이 있으면
            절단함 - 구멍 표면에서 가장 가까운 외벽을 찾아 절단선을 그은 후, Connected Components로
            분할된 각각의 조각들을 새로운 노드로 등록하는 방식임.
        Step 4: 볼록성(Solidity) 기반 오목한 공간(L자 복도 등) 분할. 이전 단계에서 생성된 노드들을
            순회하며, 볼록성이 낮은 노드에 대해 절단선을 그어 분할함 - 절단선은 구멍 탐지에서 사용한
            방식과 유사하게, 볼록성이 낮은 영역의 표면에서 가장 가까운 외벽을 찾아 그은 후,
            Connected Components로 분할된 각각의 조각들을 새로운 노드로 등록하는 방식임.
        Step 5: 너무 긴 복도를 탐지해, 일정 종횡비가 넘으면 분할함.
        그리고 마지막으로 분할된 모든 노드를 .npz 파일로 저장함(final_topological_map.npz).
        """
        print("--- [Step 3] Space Segmentation---")
        try:
            segmenter = SpaceSegmenter(
                mask=raw_mask, 
                resolution=map_resolution, 
                origin=map_origin,
                visualization_dir=self.visualization_dir,
                config=self.cfg
            )
            
            final_nodes = segmenter.execute_segmentation()
            segmenter.save_topology(final_nodes, output_dir=self.topology_dir)
            
        except Exception as e:
            print(f"[!] Space Segmentation Error: {e}")
            traceback.print_exc()
            return False

        print("[+] Environment modeling workflow finished successfully.")
        return True
