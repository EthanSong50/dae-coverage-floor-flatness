#!/usr/bin/env python3
"""
surface_profiling/reprocess_pcd.py

이미 수집되어 저장된 combined_*.pcd 파일 하나를 입력받아,
[Stage 2/3] Floor Extraction 과 [Stage 3/3] Heatmap Generation을
surface_profiler.py의 run()과 동일한 함수(extract_floor_by_height,
generate_floor_heatmap)로 다시 실행함.

목적: z_min/z_max, grid_size, map_yaml_dir 같은 params.yaml 값을
바꿔가며 히트맵 결과를 반복 확인할 때, 매번 로봇을 다시 주행시킬
필요 없이 이미 모아둔 pcd로 빠르게 재테스트하기 위함.

사용법:
    python3 reprocess_pcd.py combined_2026-07-10_23-54-47.pcd

    # params.yaml 값을 무시하고 이번 실행에만 다른 값을 쓰고 싶을 때:
    python3 reprocess_pcd.py combined_2026-07-10_23-54-47.pcd --z-min -0.01 --z-max 0.04
    python3 reprocess_pcd.py combined_2026-07-10_23-54-47.pcd --grid-size 0.01
    python3 reprocess_pcd.py combined_2026-07-10_23-54-47.pcd --no-map-overlay

주의: 이 스크립트는 params.yaml을 surface_profiler.py와 완전히 동일한
방식(ament_index_python 우선, 실패 시 상대경로 폴백)으로 읽음.
즉 "params.yaml을 고쳤는데 반영이 안 되는 것 같다"는 의심이 들 때,
이 스크립트로 재실행해서 나오는 값(아래 [*] Config values 출력)이
곧 실제 파이프라인이 쓰는 값과 100% 동일함 — 값이 여기서도 예전
그대로라면 params.yaml 수정 자체가 반영 안 된 것이고(경로 오타,
colcon build 누락 등), 여기서는 새 값인데 실제 주행 결과만 예전
같다면 그건 다른 원인(캐시된 install 바이너리 실행 등)임.
"""

import os
import sys
import argparse


def _add_utils_to_path():
    """surface_profiler.py와 동일한 방식으로 utils 모듈을 임포트 가능하게 함.
    이 스크립트가 소스 트리(src/.../surface_profiling/) 안에서 실행되는 경우와
    install 트리(install/.../surface_profiling/) 안에서 실행되는 경우 둘 다 대응."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, this_dir)


_add_utils_to_path()

try:
    from utils.floor_extractor import extract_floor_by_height
    from utils.heatmap_generator import generate_floor_heatmap
    from utils.config_paths import (
        load_config,
        resolve_pointcloud_dir,
        resolve_visualization_dir,
        resolve_map_yaml_path,
    )
except ImportError:
    from surface_profiling.utils.floor_extractor import extract_floor_by_height
    from surface_profiling.utils.heatmap_generator import generate_floor_heatmap
    from surface_profiling.utils.config_paths import (
        load_config,
        resolve_pointcloud_dir,
        resolve_visualization_dir,
        resolve_map_yaml_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="기존 combined_*.pcd를 재필터링/재시각화하는 도구 (재주행 불필요)"
    )
    parser.add_argument(
        "pcd_filename",
        help="pointcloud_dir 안에 있는 pcd 파일명 (예: combined_2026-07-10_23-54-47.pcd). "
             "절대/상대 경로를 직접 줘도 됨.",
    )
    parser.add_argument("--z-min", type=float, default=None, help="params.yaml의 z_min을 이번 실행에서만 덮어씀 (m)")
    parser.add_argument("--z-max", type=float, default=None, help="params.yaml의 z_max를 이번 실행에서만 덮어씀 (m)")
    parser.add_argument("--grid-size", type=float, default=None, help="params.yaml의 grid_size를 이번 실행에서만 덮어씀 (m)")
    parser.add_argument(
        "--no-map-overlay", action="store_true",
        help="params.yaml에 map_yaml_dir가 있어도 이번 실행은 맵 오버레이 없이 히트맵만 단독 생성"
    )
    parser.add_argument(
        "--map-yaml-dir", type=str, default=None,
        help="params.yaml의 map_yaml_dir를 이번 실행에서만 덮어씀"
    )
    args = parser.parse_args()

    workspace_root, profiling_cfg = load_config()

    pointcloud_dir = resolve_pointcloud_dir(workspace_root, profiling_cfg)
    visualization_dir = resolve_visualization_dir(workspace_root, profiling_cfg)

    # 입력 파일 경로 해석: 절대/상대경로로 이미 존재하면 그대로, 아니면 pointcloud_dir 기준
    pcd_path = args.pcd_filename
    if not os.path.isabs(pcd_path) and not os.path.exists(pcd_path):
        pcd_path = os.path.join(pointcloud_dir, args.pcd_filename)
    if not os.path.exists(pcd_path):
        print(f"[!] PCD file not found: {pcd_path}")
        sys.exit(1)

    z_min = args.z_min if args.z_min is not None else profiling_cfg.get('z_min', -0.005)
    z_max = args.z_max if args.z_max is not None else profiling_cfg.get('z_max', 0.035)
    grid_size = args.grid_size if args.grid_size is not None else profiling_cfg.get('grid_size', 0.02)

    if args.no_map_overlay:
        map_yaml_path = None
    elif args.map_yaml_dir is not None:
        # --map-yaml-dir로 이번 실행만 다른 폴더를 지정한 경우, 같은 조합 규칙
        # (폴더 + map_from_dae.yaml)을 적용하기 위해 profiling_cfg를 복사해서 덮어씀.
        override_cfg = dict(profiling_cfg)
        override_cfg['map_yaml_dir'] = args.map_yaml_dir
        map_yaml_path = resolve_map_yaml_path(workspace_root, override_cfg)
    else:
        # [핵심] surface_profiler.py의 _resolve_directories()와 완전히 동일한
        # resolve_map_yaml_path()를 그대로 사용 -> 두 스크립트가 절대 어긋나지 않음.
        map_yaml_path = resolve_map_yaml_path(workspace_root, profiling_cfg)
        if map_yaml_path is not None and not os.path.exists(map_yaml_path):
            print(f"[!] Warning: map_yaml_path가 설정되었으나 파일을 찾을 수 없습니다: {map_yaml_path}")

    print("[*] Config values in use:")
    print(f"    z_min          = {z_min}  ({z_min*100:.1f} cm)")
    print(f"    z_max          = {z_max}  ({z_max*100:.1f} cm)")
    print(f"    grid_size      = {grid_size}")
    print(f"    map_yaml_path  = {map_yaml_path}")
    print(f"    input pcd      = {pcd_path}")

    # 원본 파일명에서 타임스탬프/suffix를 그대로 살려, 재처리본임을 구분할 수 있는
    # 별도 파일명으로 저장함(원본 raw pcd는 덮어쓰지 않음).
    base = os.path.splitext(os.path.basename(pcd_path))[0]
    if base.startswith("combined_"):
        stem = base[len("combined_"):]
    else:
        stem = base
    filtered_filename = f"combined_filtered_{stem}_reprocessed.pcd"
    heatmap_filename = f"floor_heatmap_{stem}_reprocessed.png"

    filtered_path = os.path.join(pointcloud_dir, filtered_filename)
    heatmap_path = os.path.join(visualization_dir, heatmap_filename)

    print("\n[Stage 2/2*] Floor Extraction (Height-based Filtering)")
    extract_floor_by_height(pcd_path, filtered_path, z_min=z_min, z_max=z_max)

    print("\n[Stage 2/2*] Floor Flatness Heatmap Generation")
    generate_floor_heatmap(
        filtered_path,
        heatmap_path,
        grid_size=grid_size,
        z_min=z_min,
        z_max=z_max,
        map_yaml_dir=map_yaml_path,
    )

    print("\n=======================================================")
    print("[+] Reprocessing complete.")
    print(f"    -> Filtered PCD: {filtered_path}")
    print(f"    -> Heatmap Image: {heatmap_path}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()