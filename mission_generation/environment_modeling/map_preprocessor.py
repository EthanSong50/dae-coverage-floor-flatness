import cv2
import numpy as np
import yaml
import os

class MapPreprocessor:
    def __init__(self, yaml_path):
        """
        ROS 맵 메타데이터(.yaml파일) 로드하고 초기화.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"[!] YAML file not found: {yaml_path}")

        with open(yaml_path, 'r') as f:
            self.map_info = yaml.safe_load(f)
        
        # 경로 및 물리 파라미터 설정
        self.base_path = os.path.dirname(os.path.abspath(yaml_path))
        self.pgm_path = os.path.join(self.base_path, self.map_info['image'])
        self.resolution = float(self.map_info['resolution'])
        self.origin = self.map_info['origin']  # [x, y, yaw]

        print(f"[*] Map Preprocessor Initialized: {self.map_info['image']}")

    def get_largest_connected_area(self):
        img = cv2.imread(self.pgm_path, cv2.IMREAD_GRAYSCALE)
        if img is None: return None, None

        # 1. 이진화: 흰색 배경(255), 검은색 벽 선(0)
        _, binary = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY)
        h, w = binary.shape

        # 2. 외곽(집 바깥) 지우기 (Flood Fill)
        # 이미지의 네 모서리(여백)에서 검은색(0)을 쏟아부음.
        # 벽(0)에 닿으면 멈추므로 집 안은 보호됨.
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # 네 귀퉁이에서 시도 (하나라도 집 바깥이면 작동)
        points = [(0, 0), (0, h-1), (w-1, 0), (w-1, h-1)]
        temp_binary = binary.copy()
        
        for pt in points:
            if temp_binary[pt[1], pt[0]] == 255:
                # 외곽 여백(255)을 회색(127)으로 잠시 채움
                cv2.floodFill(temp_binary, flood_mask, pt, 127)

        # 3. 실내 영역만 추출
        # 회색(127)은 바깥, 흰색(255)은 실내 바닥임.
        indoor_only = np.zeros_like(img)
        indoor_only[temp_binary == 255] = 255

        # 4. (보강) 혹시 모를 파편 제거를 위해 다시 한번 Largest 영역 추출
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(indoor_only, connectivity=8)
        if num_labels <= 1:
            print("[!] No indoor space found after clearing exterior.")
            return None, None

        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        
        final_mask = np.zeros_like(img)
        final_mask[labels == largest_label] = 255

        print(f" -> House exterior cleared via FloodFill.")
        print(f" -> Indoor floor extraction complete. (Area: {stats[largest_label, cv2.CC_STAT_AREA]} px)")
        
        return final_mask, stats[largest_label]
        
    def save_analysis_result(self, mask, stats, visualization_dir=None, data_dir=None):
        img_save_path = os.path.abspath(visualization_dir)
        data_save_path = os.path.abspath(data_dir)

        # 3. 파일 경로
        floor_filename = os.path.join(img_save_path, "pre_filtered_floor.png")
        npz_filename = os.path.join(data_save_path, "map_preprocessed_data.npz")
        
        # 4. 저장
        cv2.imwrite(floor_filename, mask)

        np.savez(
            npz_filename,
            mask=mask,
            stats=stats,
            resolution=self.resolution,
            origin=self.origin
        )

        print(f" -> [Saved] Visualization Image: {floor_filename}")
        print(f" -> [Saved] NPZ Data: {npz_filename}")
