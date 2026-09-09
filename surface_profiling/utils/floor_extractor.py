# surface_profiling/utils/floor_extractor.py

import os
import open3d as o3d
import numpy as np


def extract_floor_by_height(pcd_path, output_path, z_min=-0.005, z_max=0.035, z_threshold=None):
    """
    PCD 파일에서 z값(높이)이 [z_min, z_max] 범위 안에 있는 포인트만 남겨
    바닥면 후보 데이터를 추출함. 벽면/장애물 형상을 분석하는 정밀 필터링이
    아닌, 단순 높이 밴드 컷오프 기반 필터링임.

    z_min, z_max: 바닥으로 간주할 높이 범위(m). 기본값은 바닥 요철이
        일반적으로 수 cm 이내인 것을 감안한 값이며, 실측 데이터에 맞춰
        조정 필요.
    z_threshold: (하위 호환용, deprecated) 상한만 지정하고 싶을 때 사용.
        지정되면 z_max를 덮어쓰고 z_min은 무시(-inf로 취급)함(상한
        컷오프만 적용하는 동작).
    """
    if not os.path.exists(pcd_path):
        raise FileNotFoundError(f"[-] PCD file not found: {pcd_path}")

    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points)

    if z_threshold is not None:
        # 하위 호환 모드: 상한 컷오프만 적용(벽 하단이 섞여 들어올 수 있음)
        mask = points[:, 2] <= z_threshold
    else:
        mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)

    filtered_points = points[mask]

    pcd_filtered = o3d.geometry.PointCloud()
    pcd_filtered.points = o3d.utility.Vector3dVector(filtered_points)

    o3d.io.write_point_cloud(output_path, pcd_filtered)

    print(f"[*] 원본 포인트: {len(points)}, 필터링 후 포인트: {len(filtered_points)}")
    print(f"[+] Saved floor-filtered PCD: {output_path}")
    return output_path