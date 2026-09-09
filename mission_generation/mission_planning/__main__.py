# mission_generation/mission_planning/__main__.py

import os
import sys
import yaml
import json
import argparse
import traceback

from ament_index_python.packages import get_package_share_directory

try:
    from .mission_planning import MissionPlanner
except ImportError:
    from mission_planning import MissionPlanner

def main():
    parser = argparse.ArgumentParser(description="Mission Planner Standalone Core Runner")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None, 
        help="Explicit path to params.yaml configuration file"
    )
    args = parser.parse_args()

    if args.config:
        config_path = os.path.abspath(args.config)
    else:
        try:
            package_share_dir = get_package_share_directory('dae_coverage_floor_flatness')
            config_path = os.path.join(package_share_dir, 'config', 'params.yaml')
        except Exception:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            config_path = os.path.join(base_dir, "config", "params.yaml")


    print("\n" + "="*60)
    print("[*] Mission Planner Sub-Package Core Standalone Module")
    print("="*60)
    print(f"[*] Target Config Resource: {config_path}")

    if not os.path.exists(config_path):
        print(f"[!] Critical Error: Global parameter registry missing at {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    global_cfg = config.get('global', {})
    workspace_root = os.path.expanduser(global_cfg.get('workspace_root', '~/dae_floor_maps'))

    env_cfg = config.get('environment_modeling', {})
    topology_dir = os.path.join(workspace_root, env_cfg.get('output_topology_dir', 'maps/topology'))
    map_file = os.path.normpath(os.path.join(topology_dir, "final_topological_map.npz"))

    if not os.path.exists(map_file):
        print(f"[!] Dependency Missing: Preprocessed topological map asset not found at:\n    {map_file}")
        print("[!] Please execute map environment pre-processing before planning.")
        sys.exit(1)

    mission_cfg = config.get('mission_planner', {})
    if not mission_cfg:
        print("[!] Warning: 'mission_planner' section missing in configuration. Using system defaults.")
        mission_cfg = {}

    planner_cfg = mission_cfg.copy()
    # MissionPlanner.__init__()이 받지 않는 키(plan() 단계 전용 파라미터, 경로 설정용 키)는
    # 반드시 모두 pop해서 **planner_cfg로 생성자에 전달되지 않도록 분리함 - 안 그러면
    # "unexpected keyword argument" TypeError가 발생함. 현재 params.yaml의
    # mission_planner 섹션에는 이 두 키만 존재하지만, 과거 버전의 sampling_step처럼
    # 나중에 안 쓰는 키가 남아있을 수 있으니 새 kwarg-only 설정을 추가할 때 이 목록도
    # 함께 갱신할 것.
    planner_vis_rel = planner_cfg.pop('visualization_dir', 'visualization/mission_generation/mission_planning')
    planner_vis_path = os.path.join(workspace_root, planner_vis_rel)

    metric_rel = planner_cfg.pop('output_metric_dir', 'analytics/metrics')
    metric_dir = os.path.join(workspace_root, metric_rel)
    cache_file = os.path.normpath(os.path.join(metric_dir, "final_path.json"))

    print(f"[*] Map Data Asset: {map_file}")
    print(f"[*] Output Waypoint Destination: {cache_file}")

    # 4. MissionPlanner
    try:
        planner = MissionPlanner(
            topomap_path=map_file,
            visualization_dir=planner_vis_path,
            **planner_cfg
        )

        final_path = planner.plan(
            save_debug=True,
            show_plot=False,
            output_dir=metric_dir
        )
        
        if final_path:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(final_path, f, indent=4)
            print(f"\n[+] Standalone Operations Completed Successfully!")
            print(f"    - Destination Target: {cache_file}")
            print(f"    - Total Nav2 Waypoints Cached: {len(final_path)}")
        else:
            print("[!] Core Engine Failure: Path translation returned an empty matrix.")
            sys.exit(1)

    except Exception as e:
        print(f"\n[!] Critical Core Engine Exception Encountered: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
