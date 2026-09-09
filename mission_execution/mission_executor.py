# dae_coverage_floor_flatness/mission_execution/mission_executor.py

import os
import sys
import yaml
import json
import csv
import copy
import time
import math
import traceback

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import SingleThreadedExecutor
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult

# 경량 유틸리티 모듈
try:
    from utils.map_utils import get_map_bounds
    from utils.ros_utils import create_pose_stamped, teleport_gazebo_entity
    from utils.nav2_utils import apply_nav2_monkey_patches
    from utils.visualizer import visualize_paths, visualize_planned_wall_proximity, visualize_stall_points
    from utils.boundary_repass import BoundaryRepassController
    from utils.stall_logger import StallWatcher, write_stall_report
except ImportError:
    from mission_execution.utils.map_utils import get_map_bounds
    from mission_execution.utils.ros_utils import create_pose_stamped, teleport_gazebo_entity
    from mission_execution.utils.nav2_utils import apply_nav2_monkey_patches
    from mission_execution.utils.visualizer import visualize_paths, visualize_planned_wall_proximity, visualize_stall_points
    from mission_execution.utils.boundary_repass import BoundaryRepassController
    from mission_execution.utils.stall_logger import StallWatcher, write_stall_report


class MissionExecutor(Node):
    """
    Nav2 기반 미션 실행을 담당하는 ROS 2 노드.

    is_sim 여부는 ROS 2 파라미터(launch argument)로 주입받음. 예:
    ros2 launch dae_coverage_floor_flatness mission_execution.launch.py is_sim:=true

    주행 방식: final_path.json 전체를 방향(heading)이 바뀌는 지점 기준으로만
    재분할함(_split_into_straight_subsegments). coverage/transit 구분 없이
    모든 직선 sub-segment에서 동일하게 처리함(_execute_capture_subsegment):
    제자리 회전(Spin, 절대각 차이 기반) -> 캡처가 아직 꺼져 있으면 즉시 캡처
    시작 신호 -> 구간 끝점까지 여러 점을 한 번에 통과하는 직선
    주행(NavigateThroughPoses/goThroughPoses)하며 계속 캡처 -> 도착 시 정지.
    coverage뿐 아니라 transit 구간도 연속으로 측정해야 라이다 blind zone이
    충분히 메꿔지므로 두 종류를 다르게 처리하지 않음(자세한 배경은
    HISTORY.md §3 참고).

    캡처 종료는 record_pcd=True 구간(coverage)이 끝나는 시점, 즉 매
    "coverage exit" 경계마다 boundary_repass.BoundaryRepassController.
    run_exit_repass()가 담당함: 도착 -> 유턴 -> 왔던 방향으로 일정 거리
    되짚어 재통과(캡처 유지) -> 캡처 종료. 그 지점을 실제로 양방향(접근+
    멀어짐)으로 지나쳐야 얕은 각도에서 라이다 blind zone이 메워지는데, 정상
    도착만으로는 접근 방향 하나만 확보되므로 되짚기로 반대 방향 시야를
    인위적으로 만듦. 되짚기가 끝나면 로봇은 원래 coverage 종료 지점보다
    조금 안쪽에 있게 되지만, 그 다음 sub-segment(transit)의 정상 주행이
    로봇의 현재 위치에서부터 알아서 경로를 짜므로 별도 복귀 동작은 필요
    없음. execute_mission() 참고, 도입 경위는 HISTORY.md §1 참고.
    """

    def __init__(self):
        super().__init__('mission_executor_node')

        print("\n=======================================================")
        print("[*] Nav2 Mission Executor (Jetson / Sim Controller)")
        print("=======================================================\n")

        # self.spin_executor
        # self(MissionExecutor) 전용 SingleThreadedExecutor.
        # navigator(BasicNavigator)는 nav2_simple_commander 라이브러리 내부에서
        # 자체적으로 rclpy.spin_once(navigator, ...)를 호출하므로, 이중 spin 충돌을
        # 피하기 위해 별도 executor에 등록하지 않고 독립적으로 관리함.
        self.spin_executor = SingleThreadedExecutor()
        self.spin_executor.add_node(self)

        # 상태 변수 초기화
        self.config = None
        self.global_cfg = None
        self.env_cfg = None
        self.mission_exec_cfg = None
        self.workspace_root = None
        self.map_bounds = None
        self.final_path = None
        self.navigator = None

        # FollowWaypoints는 coverage 지점마다 여러 세그먼트(goal)로 나뉘어 전송되므로,
        # 미션 전체의 성공 여부는 navigator.getResult()(마지막 세그먼트만 반영)가 아닌
        # 이 플래그로 별도 추적함.
        self.mission_succeeded = False

        # 캡처 상태 추적. coverage(record_pcd=True) sub-segment가 끝나는
        # 시점마다 execute_mission()이 boundary_repass.run_exit_repass()를 불러
        # 되짚기 후 여기서 명시적으로 끔 - _execute_capture_subsegment 자체는
        # 캡처를 스스로 끄지 않음.
        self._capture_active = False

        # 미션 전체의 진짜 첫/마지막 지점은 "지나쳐야 채워진다" 원칙의 전제(양방향
        # 통과)를 구조적으로 만족할 수 없음 - boundary_repass.py 모듈 docstring 참고.
        self._boundary_repass = BoundaryRepassController(self)

        self._current_angular_mode = None
        self._controller_param_client = None  # 지연 생성이라 None으로 시작함

        # 주행 모니터링 관련 상태
        self.current_amcl_x = None
        self.current_amcl_y = None
        self.last_valid_amcl_pose = None
        self.max_allowed_jump = 0.60

        # 연속 점프 감지용 상태. 단발성 AMCL 보정(긴 직선 주행 후 누적된
        # dead-reckoning 오차가 한 번에 정상화되는 경우 등)까지 비상 상황으로
        # 취급해 미션을 중단시키면 너무 과민하므로, 짧은 시간 안에 여러 번
        # 반복될 때만 진짜 비상(로컬라이제이션 붕괴/텔레포트)으로 간주함.
        self.amcl_jump_timestamps = []
        self.amcl_jump_window_sec = 5.0     # 이 시간 안에
        self.amcl_jump_count_threshold = 3  # 이만큼 반복되면 비상으로 격상함
        self.path_history = []
        self._last_record_time = 0.0
        self.sub_amcl_check = None

        # 주행 중 5초 이상 진행이 멈추는 구간을 디버깅용으로 별도 기록함
        # (얼마나/왜 - number_of_recoveries 변화로 nav2 recovery 발동 여부까지
        # 함께 남김). execute_mission()의 모든 blocking 대기 루프에서 공용으로
        # 쌓이며, _save_mission_results()에서 이것만 따로 파일에 출력함.
        self._stall_events = []
        self._mission_start_wall_time = None

        # AMCL 초기화 검증 관련 상태
        self.verified_amcl_x = None
        self.verified_amcl_y = None
        self.initial_pose = None
        self.sub_verify = None

        # surface_profiling(노트북) 측 PCD 수집 종료를 알리는 서비스 클라이언트.
        # 정상 종료와 비정상(실패/타임아웃) 종료를 서로 다른 서비스로 분리해서 호출함.
        self.stop_collection_success_client = self.create_client(
            Trigger, '/surface_profiling/stop_collection_success'
        )
        self.stop_collection_abort_client = self.create_client(
            Trigger, '/surface_profiling/stop_collection_abort'
        )

        # surface_profiling(노트북) 측 지점별(Waypoint) 캡처 시작/종료 서비스 클라이언트.
        # coverage 웨이포인트에 정지할 때마다 한 쌍(start -> stop)씩 호출함.
        self.start_capture_client = self.create_client(
            Trigger, '/surface_profiling/start_waypoint_capture'
        )
        self.stop_capture_client = self.create_client(
            Trigger, '/surface_profiling/stop_waypoint_capture'
        )

        # 1. ROS 2 파라미터로 시뮬레이션 모드 여부 확보 (launch argument로 주입됨)
        self._resolve_run_mode()

        # 2. params.yaml 설정 파일 파싱
        self._load_config()

        # 3. 글로벌 맵 바운즈 확보
        self._load_map_bounds()

        # 4. 이전 파이프라인에서 생성된, 보간/샘플링이 끝난 JSON 원본 경로 파일 로드
        self._load_final_path()

    # ------------------------------------------------------------------
    # 초기화 단계 헬퍼
    # ------------------------------------------------------------------

    def _resolve_run_mode(self):
        """
        'is_sim' ROS 2 파라미터를 선언하고 읽음. launch 파일이 주입하지 않으면
        기본값 False(Real-world)로 동작함.

        use_sim_time은 launch 파일이 is_sim과 함께 주입하는 것이 표준이지만,
        혹시 누락되더라도 안전하게 동작하도록 is_sim 값으로부터 use_sim_time을
        다시 추론해 자기 자신에게 강제 적용함.
        """
        self.declare_parameter('is_sim', False)
        self.is_sim = self.get_parameter('is_sim').get_parameter_value().bool_value

        if self.is_sim:
            self.get_logger().info("Running Simulation(Gazebo) Mode.")
        else:
            self.get_logger().info("Running Real-world Mode.")

        # use_sim_time 강제 동기화 (launch 파일이 누락했을 경우의 안전장치)
        use_sim_time_param = self.get_parameter_or(
            'use_sim_time', Parameter('use_sim_time', Parameter.Type.BOOL, self.is_sim)
        )
        if use_sim_time_param.value != self.is_sim:
            self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, self.is_sim)])

    def _load_config(self):
        try:
            from ament_index_python.packages import get_package_share_directory
            package_share_dir = get_package_share_directory('dae_coverage_floor_flatness')
            config_path = os.path.join(package_share_dir, 'config', 'params.yaml')
        except Exception:
            # mission_execution 내에서 실행 시 프로젝트 루트의 config로 fallback 탐색
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            config_path = os.path.join(base_dir, "config", "params.yaml")

        print(f"[*] Resolving parameters from: {config_path}")
        if not os.path.exists(config_path):
            print(f"[!] Critical Error: params.yaml file not found at {config_path}")
            sys.exit(1)

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.global_cfg = self.config.get('global', {})
        self.workspace_root = os.path.expanduser(
            self.global_cfg.get('workspace_root', '~/dae_floor_maps')
        )
        self.env_cfg = self.config.get('environment_modeling', {})
        self.mission_exec_cfg = self.config.get('mission_execution', {})
        # 실행 시점에 직접 쓰이진 않지만, 계획 시점에 쓰인 값(final_path_meta.json)과
        # 대조하기 위해서만 참조함 - _verify_plan_meta 참고.
        self.mission_planner_cfg = self.config.get('mission_planner', {})

    def _load_map_bounds(self):
        grid_dir = os.path.join(self.workspace_root, self.env_cfg.get('output_grid_dir', 'maps/grid'))
        yaml_path = os.path.normpath(os.path.join(grid_dir, "map_from_dae.yaml"))
        self.map_yaml_path = yaml_path

        if not os.path.exists(yaml_path):
            print(f"[!] CRITICAL ERROR: Map bounds data '{yaml_path}' not found!")
            sys.exit(1)

        self.map_bounds = get_map_bounds(yaml_path)

    def _load_final_path(self):
        metric_dir = os.path.join(self.workspace_root, self.mission_exec_cfg.get('input_metric_dir', 'analytics/metrics'))
        cache_file = os.path.normpath(os.path.join(metric_dir, "final_path.json"))

        if not os.path.exists(cache_file):
            print(f"[!] CRITICAL ERROR: Mission route '{cache_file}' not found!")
            print("[-] Please run 'run_generation_pipeline.py' on your workstation first.")
            sys.exit(1)

        print(f"[*] Found pre-generated path at {cache_file}. Loading...")
        with open(cache_file, 'r') as f:
            self.final_path = json.load(f)
        print(f"[*] Path loaded. Total raw waypoints: {len(self.final_path)}")

        self._verify_plan_meta(metric_dir)

    def _verify_plan_meta(self, metric_dir):
        """
        final_path.json이 계획 시점(run_generation_pipeline.py -> MissionPlanner.plan())에
        실제로 사용한 파라미터 값을, 지금 이 노드가 params.yaml에서 읽은 실행 시점 값과
        대조함. mission_planner.py가 plan() 마지막에 함께 저장하는 사이드카
        'final_path_meta.json'을 읽어 비교함.

        boundary_repass_distance_m/enable_boundary_repass는 final_path.json의
        좌표 자체(transit이 실제로 시작하는 지점)에 기하학적으로 반영되므로, 두
        시점의 값이 어긋나면 계획된 transit 시작점과 실제 repass 후 로봇 위치가
        조용히 달라짐 - robot_width/path_safety_margin도 경로 형상 자체에
        반영되는 같은 범주의 값임. 이 일치를 사람이 매번 기억할 필요 없도록
        여기서 자동으로 대조하고, 어긋나면 다른 CRITICAL ERROR들과 동일하게
        즉시 중단시킴(도입 배경은 HISTORY.md §2 참고).

        사이드카 파일이 없으면(예: 이 검증 로직 추가 이전에 생성된 오래된
        final_path.json) 대조 자체를 건너뛰고 경고만 남김 - 하위 호환을 위해
        미션을 막지는 않음.
        """
        meta_file = os.path.normpath(os.path.join(metric_dir, "final_path_meta.json"))
        if not os.path.exists(meta_file):
            print(f"[!] Warning: '{meta_file}' not found - cannot verify plan/runtime "
                  f"parameter consistency (older final_path.json?). Proceeding without check.")
            return

        with open(meta_file, 'r') as f:
            plan_meta = json.load(f)

        runtime_values = {
            'robot_width': self.env_cfg.get('robot_width', 0.28),
            'path_safety_margin': self.mission_planner_cfg.get('path_safety_margin', 0.20),
            'boundary_repass_distance_m': self.mission_exec_cfg.get('boundary_repass_distance_m', 1.5),
            'enable_boundary_repass': self.mission_exec_cfg.get('enable_boundary_repass', True),
        }

        mismatches = []
        for key, runtime_val in runtime_values.items():
            plan_val = plan_meta.get(key)
            if plan_val is None:
                continue
            if isinstance(plan_val, bool) or isinstance(runtime_val, bool):
                mismatch = bool(plan_val) != bool(runtime_val)
            else:
                mismatch = abs(float(plan_val) - float(runtime_val)) > 1e-6
            if mismatch:
                mismatches.append((key, plan_val, runtime_val))

        if mismatches:
            print(f"\n[!!! CRITICAL ERROR !!!] final_path.json was planned with different "
                  f"parameters than this executor's current params.yaml:")
            for key, plan_val, runtime_val in mismatches:
                print(f"    - {key}: planned={plan_val}, current params.yaml={runtime_val}")
            print("[-] The planned transit start points / path geometry no longer match what "
                  "this executor would produce. Either revert params.yaml to the planned values, "
                  "or re-run run_generation_pipeline.py to regenerate final_path.json before "
                  "executing this mission.")
            sys.exit(1)

        print("[+] Plan/runtime parameter consistency verified (boundary_repass, robot_width, "
              "path_safety_margin match final_path_meta.json).")

    # ------------------------------------------------------------------
    # ROS 환경 셋업
    # ------------------------------------------------------------------

    def _setup_ros_environment(self):
        """
        BasicNavigator 인스턴스 생성, Gazebo 환경일 경우 초기 텔레포트 인터페이스 수행.

        rclpy.init()은 이 메서드에서 호출하지 않음. 이 노드(self) 자신을 포함한
        전체 ROS 2 컨텍스트는 main()에서 이미 단일하게 초기화되어 있다고 가정함.
        """
        print("[*] Connecting to Nav2 Server...")

        self.navigator = BasicNavigator()

        # BasicNavigator 레벨 몽키 패치.
        # navigator 인자는 호출 시점 일관성을 위해 받지만, 패치는 클래스 자체에 적용되므로
        # 인스턴스 유무와 무관하게 한 번만 적용되면 모든 BasicNavigator 인스턴스에 영향을 줌.
        apply_nav2_monkey_patches(self.navigator)

        # 미션이 실제로 시작해야 하는 물리적 지점 계산(_compute_mission_start_pose
        # 참고 - boundary_repass가 켜져 있으면 진짜 coverage 시작점이 아니라
        # 그보다 진행방향으로 앞선 러닝스타트 지점). sim/real 양쪽에서 이 하나의
        # 값을 그대로 재사용함(AMCL 초기 위치 힌트(_initialize_localization)와
        # 반드시 동일해야 함 - 어긋나면 AMCL이 틀린 방향을 정답이라 믿고 위치만
        # 수렴해버릴 수 있음).
        self._mission_start_pose = self._compute_mission_start_pose()

        if self.is_sim:
            temp_pose = copy.deepcopy(self._mission_start_pose)
            temp_pose.pose.position.z = 0.05

            teleport_gazebo_entity(temp_pose)
            print("[+] Gazebo Teleportation Successful (position + orientation matched to mission start pose).")
        else:
            print("[*] Real-world Mode: Skipping Gazebo Robot location Initializing.")

        if self.is_sim:
            print("[*] Waiting for Gazebo /clock synchronization...")
            while self.navigator.get_clock().now().nanoseconds == 0:
                rclpy.spin_once(self.navigator, timeout_sec=0.01)
            print("[+] Gazebo clock synchronization done.")

        print("[*] Waiting for Nav2 nodes to become fully Active...")
        self.navigator.waitUntilNav2Active()
        print("[+] Nav2 is now fully Active. Securing subscription margin...")
        time.sleep(4.0)

        # coverage run 시작 시 '제자리 회전'을 위해 현재 로봇의 실시간 헤딩(map->base_link)이
        # 필요함. /amcl_pose 토픽(저주파, AMCL 보정 시점에만 갱신)이 아니라 TF buffer를
        # 직접 조회하는 이유는, TF는 odom 기반으로 계속 보간되어 훨씬 더 실시간에 가까운
        # 값을 주기 때문임(회전 판단 시점의 정확도가 중요).
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    # ------------------------------------------------------------------
    # AMCL 초기 위치 수렴
    # ------------------------------------------------------------------

    def _target_verification_callback(self, msg):
        self.verified_amcl_x = msg.pose.pose.position.x
        self.verified_amcl_y = msg.pose.pose.position.y

    def _initialize_localization(self):
        """
        AMCL 초기 위치 수렴 스테이지 제어.
        수렴 실패/타임아웃 시 안전하게 시스템을 다운시키고 에러 코드로 종료함.

        '/amcl_pose' Subscription은 self(MissionExecutor) 노드에 생성함.
        따라서 이 단계의 spin은 self를 기준으로 수행함.
        """
        print("[*] Entering Robust AMCL Initialization Stage...")

        self.sub_verify = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._target_verification_callback, 10
        )
        self.initial_pose = self._mission_start_pose

        start_verify_time = time.time()
        is_localization_safe = False
        publish_interval = 0.5
        last_publish_time = 0.0

        print("[*] Dynamically injecting Initial Pose until AMCL responds...")
        while time.time() - start_verify_time < 20.0:
            self.spin_executor.spin_once(timeout_sec=0.05)
            current_time = time.time()

            if self.verified_amcl_x == 0.0 or self.verified_amcl_x is None:
                if current_time - last_publish_time >= publish_interval:
                    self.initial_pose.header.stamp = self.navigator.get_clock().now().to_msg()
                    self.initial_pose.pose.position.z = 0.0
                    print(f"  └─> [Pulse] Sending Initial Pose. Stamp Sec: {self.initial_pose.header.stamp.sec}")
                    self.navigator.setInitialPose(self.initial_pose)
                    last_publish_time = current_time
            else:
                dx = self.verified_amcl_x - self.initial_pose.pose.position.x
                dy = self.verified_amcl_y - self.initial_pose.pose.position.y
                error_dist = (dx ** 2 + dy ** 2) ** 0.5

                if error_dist > 0.5:
                    print(f"\n[!!! CRITICAL INITIALIZATION BLOCKED !!!] AMCL initialized in the WRONG ROOM!")
                    print(f"[-] Target: ({self.initial_pose.pose.position.x:.2f}, {self.initial_pose.pose.position.y:.2f})")
                    print(f"[-] AMCL Refused and went to: ({self.verified_amcl_x:.2f}, {self.verified_amcl_y:.2f})")
                    self.navigator.cancelTask()
                    self._notify_surface_profiling_stop(success=False, message="AMCL initialized in the wrong room.")
                    self.destroy_subscription(self.sub_verify)
                    self.spin_executor.remove_node(self)
                    self.destroy_node()
                    self.navigator.destroy_node()
                    rclpy.shutdown()
                    sys.exit(1)
                else:
                    print(f"\n[+] AMCL Successfully aligned within safe zone (Error: {error_dist:.3f}m).")
                    is_localization_safe = True
                    break
            time.sleep(0.05)

        if not is_localization_safe:
            print("[-] localization verification Failed.")
            self._notify_surface_profiling_stop(success=False, message="Localization verification timed out.")
            self.destroy_subscription(self.sub_verify)
            self.spin_executor.remove_node(self)
            self.destroy_node()
            self.navigator.destroy_node()
            rclpy.shutdown()
            sys.exit(1)

        wait_sec = self.mission_exec_cfg.get('post_localization_wait_sec', 3.0)
        if wait_sec > 0:
            print(f"[*] Holding position for {wait_sec:.1f}s to let AMCL settle...")
            wait_start = time.time()
            while time.time() - wait_start < wait_sec:
                self.spin_executor.spin_once(timeout_sec=0.05)
                time.sleep(0.05)

        if self.is_sim:
            print("[*] Clearing costmaps explicitly after simulation teleport & AMCL convergence...")
            self.navigator.clearAllCostmaps()

        self.destroy_subscription(self.sub_verify)
        print("[+] Initialization Stage Cleared. Moving to Path Sampling...")

    # ------------------------------------------------------------------
    # 경로 샘플링 (클램핑 전용 — 보간/샘플링은 mission_planner.py에서 완료됨)
    # ------------------------------------------------------------------

    def _prepare_goal_poses(self):
        target_margin = self.env_cfg.get('robot_width', 0.28) * 5

        goal_poses = []
        out_of_bounds_count = 0

        safe_bounds = {
            'min_x': self.map_bounds['min_x'] + target_margin,
            'max_x': self.map_bounds['max_x'] - target_margin,
            'min_y': self.map_bounds['min_y'] + target_margin,
            'max_y': self.map_bounds['max_y'] - target_margin
        }

        for wp_dict in self.final_path:
            pose_stamped = create_pose_stamped(self.navigator, wp_dict, safe_bounds)
            goal_poses.append(pose_stamped)

            if (abs(pose_stamped.pose.position.x - wp_dict['pose']['position']['x']) > 0.01 or
                    abs(pose_stamped.pose.position.y - wp_dict['pose']['position']['y']) > 0.01):
                out_of_bounds_count += 1

        if out_of_bounds_count > 0:
            print(f"[!] Warning: {out_of_bounds_count} waypoints were nudged into the safe map zone.")

        return goal_poses

    def _compute_mission_start_pose(self):
        """미션이 실제로 시작해야 하는 물리적 지점(로봇을 스폰/배치해야 할
        곳)을 계산해서 반환함.

        boundary_repass가 켜져 있으면, 미션의 진짜 첫 coverage 지점(p0)이
        아니라 거기서 진행방향으로 boundary_repass_distance_m만큼(첫
        sub-segment 길이의 90%로 clamp) 앞선 '러닝스타트' 지점을 반환함 -
        거기서부터 캡처를 켠 채로 p0까지 주행해 들어가는 것 자체가 미션의
        첫 동작이 됨(run_start_prepass가 이 지점에서 p0로 들어가는 동작만
        수행함, 배경은 HISTORY.md §1 참고). enable_boundary_repass=false이거나
        첫 sub-segment가 너무 짧으면 p0 그대로 반환함.

        반환값의 orientation은 p0를 향하는 방향(첫 sub-segment 진행방향의
        반대)임. sim에서는 이 pose가 그대로 Gazebo 텔레포트 좌표가 되고,
        real-world에서는 AMCL 초기 위치 힌트로 쓰이므로 실제 로봇도 이
        좌표/방향에 물리적으로 배치돼야 함 - 아래에서 명확히 출력함.
        """
        goal_poses = self._prepare_goal_poses()
        sub_segments = self._split_into_straight_subsegments(goal_poses)
        first_s, first_e = sub_segments[0]
        seg_poses = goal_poses[first_s:first_e]
        p0 = seg_poses[0]

        enabled = (
            self.mission_exec_cfg.get('enable_boundary_repass', True)
            and self.final_path[first_s]['header'].get('record_pcd', True)
        )
        runway_pose = self._boundary_repass.compute_runway_pose(seg_poses) if enabled else None

        if runway_pose is None:
            print(f"[*] Mission start pose = true coverage start point p0 "
                  f"({p0.pose.position.x:.2f}, {p0.pose.position.y:.2f}) "
                  f"(boundary repass disabled, or first segment too short for a runway).")
            return p0

        q = runway_pose.pose.orientation
        facing_deg = math.degrees(2.0 * math.atan2(q.z, q.w)) % 360
        d = math.hypot(runway_pose.pose.position.x - p0.pose.position.x,
                        runway_pose.pose.position.y - p0.pose.position.y)
        print(f"[*] Mission start pose = boundary-repass runway point "
              f"({runway_pose.pose.position.x:.2f}, {runway_pose.pose.position.y:.2f}), "
              f"facing {facing_deg:.0f}° toward the true coverage start "
              f"({p0.pose.position.x:.2f}, {p0.pose.position.y:.2f}), {d:.2f}m away.")
        if not self.is_sim:
            print("[!] REAL-WORLD: place the robot physically at this runway point/heading "
                  "BEFORE starting this executor (not at the coverage start point) - "
                  "AMCL initializes from this pose.")

        return runway_pose

    # ------------------------------------------------------------------
    # 주행 모니터링 콜백
    # ------------------------------------------------------------------

    def _amcl_monitor_callback(self, msg):
        self.current_amcl_x = msg.pose.pose.position.x
        self.current_amcl_y = msg.pose.pose.position.y
        curr_time = time.time()

        # 5Hz 샘플링 (약 0.5초 간격으로 기록)
        if curr_time - self._last_record_time >= 0.5:
            self.path_history.append([curr_time, self.current_amcl_x, self.current_amcl_y])
            self._last_record_time = curr_time

    # ------------------------------------------------------------------
    # 미션 실행 메인 루프
    # ------------------------------------------------------------------

    def execute_mission(self):
        """
        '/amcl_pose' 모니터링 Subscription은 self(MissionExecutor) 노드에 생성하며,
        self.spin_executor(SingleThreadedExecutor)로 spin함. navigator.isTaskComplete()/
        getFeedback() 등 BasicNavigator 자체의 내부 동작은 nav2_simple_commander
        라이브러리 구현상 navigator 자신을 별도로 spin해야 하므로, 모니터링 루프
        안에서는 self.executor와 navigator를 각각 독립적으로 spin함.

        방향전환 기반 통합 상태기계:
        coverage(F2C 스와스)와 transit(A* 커넥터) 모두 바닥을 연속으로 측정해야
        blind zone(라이다 최소 측정거리 사각지대)이 충분히 메꿔짐(배경은
        HISTORY.md §1/§3 참고). 전체 final_path를 "방향(heading)이 바뀌는
        지점"만 기준으로 재분할해서, 모든 직선 구간에서 동일한 패턴을
        반복함:

            제자리 회전(Spin, 직전 구간과의 절대각 차이) -> 캡처 시작(이미
            켜져 있으면 생략) -> 구간 끝점까지 직선 주행하며 계속 캡처 -> 도착.

        구간 경계는 각 웨이포인트에 이미 기록된 orientation(translator.py가
        진행방향 기준으로 계산해둔 값)을 연속 비교해서 찾음 —
        direction_change_threshold_deg를 넘는 지점마다 새 구간 시작. coverage
        세그먼트 하나(F2C 스와스)는 태생적으로 직선이라 항상 구간 하나 그대로
        유지되고, A* 경로로 꺾임이 있는 transit은 꺾이는 지점마다 자동으로
        잘게 쪼개져서 각각 정지-측정됨. 좁은 방/복도가 단일 점으로 축약되는
        경우도 이 분할 로직에서 자연스럽게 길이 1짜리 구간으로 처리되어
        별도의 특수 케이스 코드가 필요 없음.

        캡처 종료는 sub-segment가 끝나는 시점에 곧바로 일어나지 않음.
        record_pcd=True(coverage) sub-segment가 끝났고, 그 다음 sub-segment가
        없거나 record_pcd=False(transit)로 바뀌는 지점 — 즉 "coverage exit"
        경계마다, self._boundary_repass.run_exit_repass()를 불러 도착 ->
        유턴 -> 왔던 방향으로 되짚어 재통과(캡처 유지) -> 캡처 종료를
        수행함. 미션의 진짜 마지막 지점도 "다음 sub-segment가 없는" 경우로
        자연히 이 조건에 포함되므로 별도 특수 처리가 필요 없음. 되짚기가
        끝나면 로봇이 원래 exit 지점보다 약간 안쪽에 있게 되지만, 이어지는
        transit sub-segment의 주행이 로봇의 실제 현재 위치부터 경로를 새로
        짜므로 복귀 동작 없이 정상 진행됨 - 단, mission_planner.py가 매
        transit A* 경로를 항상 직전 세그먼트의 마지막 점에서부터 탐색해서
        만들기 때문에, 모든 세그먼트 경계에는 "직전 세그먼트의 마지막 점과
        좌표가 같은" 구조적 중복점이 원래부터 존재함(repass와 무관하게
        final_path.json 자체의 특성). 이 중복점은 새 목적지 정보가 없어
        항상 안전하게 건너뛸 수 있으므로, 직전 세그먼트 마지막 점과 좌표가
        같은 sub-segment 첫 점은 매번 건너뜀(루프 앞머리 참고) - 단, 이
        스킵은 record_pcd=False(transit) sub-segment로만 한정함(coverage
        sub-segment에 적용하면 코너 없는 2점짜리 스와스가 1점으로 줄어
        캡처가 통째로 빠지는 회귀가 생김). 이 스킵 로직이 왜 필요한지, 왜
        transit으로만 한정하는지의 배경은 HISTORY.md §1 참고.

        각 구간은 별도 goal로 순차 전송되므로, 미션 전체의 성공/실패는
        self.mission_succeeded 플래그로 별도 추적함(navigator.getResult()는
        마지막으로 전송된 goal 하나의 결과만 반영하기 때문).
        """
        env_text = "in Gazebo" if self.is_sim else "to Real-world Robot"
        print(f"[*] Executing Mission {env_text} (Direction-Change-Based Continuous Capture)...")

        self._mission_start_wall_time = time.time()

        goal_poses = self._prepare_goal_poses()
        sub_segments = self._split_into_straight_subsegments(goal_poses)
        print(f"[*] Total waypoints: {len(goal_poses)}, split into {len(sub_segments)} "
              f"straight sub-segments (coverage + transit measured continuously).")

        self.last_valid_amcl_pose = (self.initial_pose.pose.position.x, self.initial_pose.pose.position.y)
        self.amcl_jump_timestamps = []
        self.sub_amcl_check = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self._amcl_monitor_callback, 10
        )

        self.mission_succeeded = True

        first_s, first_e = sub_segments[0]
        if self.final_path[first_s]['header'].get('record_pcd', True):
            self._boundary_repass.run_start_prepass(goal_poses[first_s:first_e])

        for seg_idx, (s, e) in enumerate(sub_segments):
            seg_poses = goal_poses[s:e]
            seg_header = self.final_path[s]['header']
            seg_type_prefix = seg_header['task_type'].split('_')[0]
            record_pcd = seg_header.get('record_pcd', True)

            if seg_idx > 0 and not record_pcd and len(seg_poses) > 1:
                # mission_planner.py는 매 노드 진입/이탈 transit의 A* 경로를
                # 항상 current_pos(=직전 세그먼트의 마지막 점)에서부터 탐색해서
                # 만듦 - 그 결과 final_path.json에는 세그먼트 경계마다 "직전
                # 세그먼트의 마지막 점과 좌표가 완전히 같고 orientation만 다른"
                # 구조적 중복점이 항상 존재함(boundary_repass와 무관하게 원래부터
                # 있던 구조). run_exit_repass가 로봇을 그 지점보다 뒤로
                # 물려놓으면(반대 방향을 보게 됨) 이 중복점이 로봇 기준 '뒤쪽'에
                # 남아, min_vel_x=0.0(후진 불가) 상태에서 goThroughPoses가
                # 순서대로 방문하려다 즉시 FAILED가 남. 이 중복점은 새 목적지
                # 정보가 없어 repass 발동 여부와 무관하게 항상 안전하게 제거
                # 가능함 - 로봇이 이미 그 위치에 있는 via-point 방문은 no-op이기
                # 때문. 그래서 직전 sub-segment의 마지막 점과 좌표가 같은 경우
                # 항상 건너뜀.
                #
                # 이 스킵은 record_pcd=False(transit)인 경우로만 한정함.
                # coverage sub-segment의 첫 점은 이 문제와 무관함 - repass는
                # coverage exit에서만 일어나고, coverage에 새로 진입할 때는
                # 로봇이 항상 정확히 그 시작점에 있음. 여기서 record_pcd를
                # 안 가리고 스킵하면 2점짜리(코너 없는 단일 직선) coverage
                # 스와스가 1점으로 줄어 "고립된 단일 점"으로 오판되고, 스와스
                # 구간 전체의 캡처가 빠지는 회귀가 생김(발견 경위·실측은
                # HISTORY.md §1 참고).
                prev_last_pose = goal_poses[sub_segments[seg_idx - 1][1] - 1]
                if (abs(seg_poses[0].pose.position.x - prev_last_pose.pose.position.x) < 1e-3
                        and abs(seg_poses[0].pose.position.y - prev_last_pose.pose.position.y) < 1e-3):
                    seg_poses = seg_poses[1:]

            print(f"\n[>>>] Sub-segment {seg_idx + 1}/{len(sub_segments)}: "
                f"origin_type='{seg_type_prefix}', record_pcd={record_pcd}, "
                f"points={len(seg_poses)} (global idx {s}~{e - 1})")

            is_genuine_single = self.final_path[s]['header']['task_type'].endswith('_single')
            seg_label = f"seg{seg_idx + 1}/{len(sub_segments)}:{seg_type_prefix}"
            ok = self._execute_capture_subsegment(
                seg_poses, record_pcd=record_pcd, is_genuine_single=is_genuine_single,
                seg_type=seg_type_prefix, label=seg_label,
            )

            if not ok:
                print(f"[-] Sub-segment {seg_idx + 1} failed. Aborting mission.")
                self.mission_succeeded = False
                break

            # coverage exit 경계: 방금 끝난 sub-segment가 record_pcd=True였고,
            # 다음 sub-segment가 없거나(=미션 진짜 마지막 지점) record_pcd=False로
            # 바뀐다면(=다음이 transit) 여기가 exit임. 매번 여기서 되짚기 후
            # 캡처를 끔 - 뒤이은 transit sub-segment는 로봇의 실제 위치부터
            # 알아서 경로를 짜므로 별도 복귀 동작이 필요 없음(다음 sub-segment의
            # 구조적 중복 첫 점은 위 루프 앞머리에서 이미 걸러짐).
            is_last_segment = (seg_idx == len(sub_segments) - 1)
            next_record_pcd = None if is_last_segment else \
                self.final_path[sub_segments[seg_idx + 1][0]]['header'].get('record_pcd', True)
            is_coverage_exit = record_pcd and (is_last_segment or not next_record_pcd)

            if is_coverage_exit and self._capture_active:
                self._boundary_repass.run_exit_repass(seg_poses)

        if self._capture_active:
            # 루프가 break로 중단됐다면(세그먼트 실패), 로봇의 실제 위치가 계획과
            # 다를 수 있으므로 추가 주행 없이 즉시 캡처만 종료함. 정상 종료라면
            # 위 루프에서 마지막 coverage exit이 이미 repass로 캡처를 껐을 것이므로
            # 이 분기에 도달하지 않음.
            self._call_capture_service(self.stop_capture_client, "stop_waypoint_capture")
            self._capture_active = False

        if self.sub_amcl_check is not None:
            self.destroy_subscription(self.sub_amcl_check)
            self.sub_amcl_check = None

        if not self.mission_succeeded and not self.navigator.isTaskComplete():
            self.navigator.cancelTask()

    def _split_into_straight_subsegments(self, goal_poses):
        """
        전체 경로(goal_poses)를 방향(heading)이 바뀌는 지점, 그리고 record_pcd가
        바뀌는 지점마다 분할함. 각 점에 이미 기록된 orientation(진행방향 기준)을
        연속 비교해서 yaw변화량이 direction_change_threshold_deg(기본 15도)를
        넘으면 그 지점을 새 구간의 시작으로 삼음. 반환값은
        [(start_idx, end_idx_exclusive), ...].

        note: min_rotation_deg(_rotate_in_place_to에서 사용, 기본 3도)보다 이
        threshold를 더 크게 잡는 이유는, 미세한 리샘플링/부동소수점 오차로
        생기는 각도 잡음까지 매번 별도 구간(=매번 정지)으로 나눠버리면 불필요한
        정지가 과도하게 늘어나기 때문임.

        record_pcd 경계에서도 반드시 분할해야 하는 이유: narrow 버킷(스와스 1개)
        노드는 order_swaths_by_entry가 진입 방향에 맞춰 스와스 시작점을 고르기
        때문에, 진입 transit의 마지막 heading과 coverage 스와스의 heading이
        구조적으로 일치하는 경우가 흔함. heading만으로 분할하면 이 경계가 안
        잘려서 transit(record_pcd=False)과 coverage(record_pcd=True)가 한
        sub-segment로 합쳐지고, execute_mission()이 병합된 그룹의 record_pcd를
        맨 앞 점 하나로만 판단하므로 coverage 구간 전체의 캡처가 조용히
        통째로 사라짐(원인 확정 경위는 HISTORY.md §1 참고).
        """
        n = len(goal_poses)
        if n <= 1:
            return [(0, n)]

        threshold = math.radians(self.mission_exec_cfg.get('direction_change_threshold_deg', 15.0))

        segments = []
        seg_start = 0
        prev_yaw = self._quaternion_to_yaw(goal_poses[0].pose.orientation)
        prev_record_pcd = self.final_path[0]['header'].get('record_pcd', True)

        for i in range(1, n):
            curr_yaw = self._quaternion_to_yaw(goal_poses[i].pose.orientation)
            dyaw = curr_yaw - prev_yaw
            dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))  # -pi ~ +pi 정규화

            curr_record_pcd = self.final_path[i]['header'].get('record_pcd', True)

            if abs(dyaw) > threshold or curr_record_pcd != prev_record_pcd:
                segments.append((seg_start, i))
                seg_start = i

            prev_yaw = curr_yaw
            prev_record_pcd = curr_record_pcd

        segments.append((seg_start, n))

        # 코너와 코너가 바로 이웃해서 생기는 고립된 1점짜리 sub-segment(진짜 F2C 단일점 방이 '_single' 태그가 아닌 경우)는
        # 별도로 세워서 처리하지 않고 다음 sub-segment 맨 앞에 편입시킴 - 그러면 그 코너점이
        # goThroughPoses의 경유점 중 하나로 포함되어, 혼자 남겨지는 상황을 방지함.
        # 단, record_pcd가 다음 sub-segment와 다르면 병합하지 않음 - 병합하면 그 1점의 캡처
        # 여부가 이웃 세그먼트의 flag로 조용히 덮어써지기 때문.
        merged_segments = []
        i = 0
        while i < len(segments):
            s, e = segments[i]
            is_trivial_single = (
                (e - s == 1)
                and not self.final_path[s]['header']['task_type'].endswith('_single')
                and i + 1 < len(segments)
                and self.final_path[s]['header'].get('record_pcd', True)
                    == self.final_path[segments[i + 1][0]]['header'].get('record_pcd', True)
            )
            if is_trivial_single:
                _, next_e = segments[i + 1]
                merged_segments.append((s, next_e))
                i += 2
            else:
                merged_segments.append((s, e))
                i += 1

        return merged_segments

    # ------------------------------------------------------------------
    # AMCL 점프 감지 (여러 모니터링 루프에서 공용으로 재사용)
    # ------------------------------------------------------------------

    def _check_amcl_jump(self):
        """
        직전에 기록된 AMCL pose 대비 순간 이동 거리가 임계치(self.max_allowed_jump)를
        넘으면 '점프 후보'로 기록함. 하지만 단발성 점프(예: 긴 직선 구간을 도는
        동안 누적된 dead-reckoning 오차가 AMCL의 정상적인 재정렬로 한 번에 보정되는
        경우)는 실제로는 위험이 아니라 오히려 위치 추정이 더 정확해진 것이므로,
        그것만으로 미션을 중단시키지 않음. amcl_jump_window_sec 안에
        amcl_jump_count_threshold번 이상 반복될 때만 진짜 비상(로컬라이제이션
        붕괴, 텔레포트 등)으로 간주해 True를 반환함.

        중요: last_valid_amcl_pose는 점프 판정 여부와 무관하게 '항상' 현재 값으로
        갱신함 - 판정 순간에만 갱신을 건너뛰면, AMCL이 이미 새로운 위치에
        안정적으로 자리잡은 뒤에도 계속 옛날 기준점과 비교해 '같은 점프'를
        영원히 재판정하는 고착 상태가 생길 수 있음(비상 상황에서 절대 복구되지
        않고 미션이 항상 중단됨).
        """
        if self.current_amcl_x is None or self.current_amcl_y is None:
            return False

        is_emergency = False

        if self.last_valid_amcl_pose is not None:
            dx = self.current_amcl_x - self.last_valid_amcl_pose[0]
            dy = self.current_amcl_y - self.last_valid_amcl_pose[1]
            jump_distance = (dx ** 2 + dy ** 2) ** 0.5

            if jump_distance > self.max_allowed_jump:
                now = time.time()
                # 윈도우 밖으로 벗어난 오래된 기록은 버림
                self.amcl_jump_timestamps = [
                    t for t in self.amcl_jump_timestamps if now - t <= self.amcl_jump_window_sec
                ]
                self.amcl_jump_timestamps.append(now)

                if len(self.amcl_jump_timestamps) >= self.amcl_jump_count_threshold:
                    print(f"\n[!!! CRITICAL EMERGENCY !!!] AMCL jumped {len(self.amcl_jump_timestamps)} times "
                          f"within {self.amcl_jump_window_sec:.1f}s (latest: {jump_distance:.3f}m). "
                          f"Treating as localization failure.")
                    is_emergency = True
                else:
                    print(f"[*] AMCL correction observed: {jump_distance:.3f}m "
                          f"({len(self.amcl_jump_timestamps)}/{self.amcl_jump_count_threshold} within "
                          f"{self.amcl_jump_window_sec:.1f}s window). Treating as a normal re-localization, "
                          f"not aborting.")

        # 점프 판정 여부와 무관하게 항상 갱신함
        self.last_valid_amcl_pose = (self.current_amcl_x, self.current_amcl_y)

        return is_emergency

    # ------------------------------------------------------------------
    # 직선 sub-segment 실행 (회전 -> [필요시] 캡처 시작 -> 직선 주행+캡처).
    # coverage/transit 구분 없이 모든 직선 구간에 동일하게 적용됨. 캡처
    # 종료는 이 함수의 책임이 아님 - coverage exit 경계에서
    # execute_mission()이 boundary_repass.run_exit_repass()를 통해 명시적으로
    # 끔(그 안에서 되짚기 왕복까지 마친 뒤 종료).
    # ------------------------------------------------------------------

    def _execute_capture_subsegment(self, seg_poses, record_pcd=True, is_genuine_single=True, seg_type='transit', label='sub-segment'):
        self._set_angular_dist_threshold('coverage' if seg_type == 'coverage' else 'transit')
        capture_sec_single = self.mission_exec_cfg.get('active_capture_seconds', 2.0)
        end_pose = seg_poses[-1]
        is_single_point = (len(seg_poses) == 1)

        # 1. 제자리 회전
        if is_single_point and not is_genuine_single:
            if not self._navigate_to_pose_blocking(seg_poses[0], label=label):
                print("[!] Warning: failed to reach isolated corner point. Proceeding anyway.")
        elif not is_single_point:
            if not self._rotate_in_place_to(end_pose, label=label):
                print("[!] Warning: In-place rotation failed or skipped. Proceeding anyway.")

        # 2. 캡처 시작 신호.
        #    이미 캡처가 켜져 있으면(같은 노드 안에서 coverage sub-segment가
        #    연달아 이어지는 경우, 예: 스와스 중간의 90도 코너) 재시작하지 않고
        #    그대로 이어감 - 캡처 창이 끊기지 않아야 한 창에 계속 쌓임.
        started = self._capture_active

        if record_pcd and not self._capture_active:
            started = self._call_capture_service(self.start_capture_client, "start_waypoint_capture")  # 측정 누락 방지를 위해 캡처 신호를 주행보다 먼저 보냄
            if started:
                # 정지 대기는 넣지 않음 - 가시 링만 진해질 뿐 사각지대는 그대로이므로
                # 캡처 시작 직후 바로 이동함.
                print("  [Capture] Capture started, moving immediately (no stationary settle).")
            else:
                print("[!] Skipping this sub-segment's capture window (start signal failed).")
            self._capture_active = started
        elif not record_pcd:
            print("  [Capture] record_pcd=False — skipping capture, driving through.")

        # 3. 주행
        ok = True
        if not is_single_point:
            print(f"  [Drive] Straight sub-segment: {len(seg_poses)} points "
                  f"({'continuous capture' if self._capture_active else 'no capture'} while moving).")
            ok = self._navigate_through_poses_blocking(seg_poses, label=label)
        elif started:
            print("  [Drive] Single-point sub-segment: staying in place for capture.")
            self._spin_sleep(capture_sec_single)
        elif record_pcd:
            print("  [Drive] Single-point sub-segment: capture start failed, skipping dwell.")
        else:
            print("  [Drive] Single-point sub-segment: record_pcd=False, skipping dwell entirely.")

        return ok

    # ------------------------------------------------------------------
    # 제자리 회전 (nav2_msgs/action/Spin, 절대각 차이를 상대 회전량으로 변환)
    # ------------------------------------------------------------------

    def _quaternion_to_yaw(self, q):
        # 평면 회전만 다루므로 x=y=0을 가정하고 z, w만으로 yaw를 계산함.
        return 2.0 * math.atan2(q.z, q.w)

    def _lookup_base_link_transform(self):
        """
        map->base_link TF를 조회함. self.spin_executor는 백그라운드 스레드 없이
        여러 blocking 루프 안에서 수동으로 spin_once되는 방식이라(예:
        _rotate_in_place_to의 대기 루프), tf2_ros.Buffer의 built-in timeout
        (콜백 알림으로 깨어나는 방식)이 여기서는 작동하지 않음 - 이 호출
        스레드가 곧 유일한 spin 주체라서, timeout만 넘기면 그 시간 동안 아무
        콜백도 못 돌고 그냥 실패함.

        run_start_prepass가 첫 액션으로 이 조회를 호출하는데, AMCL 수렴 직후
        시점과 매우 가까워서 self.tf_buffer/tf_listener가 생성된 지 얼마 안 돼
        버퍼에 이 조회 시각까지의 이력이 아직 없어 ExtrapolationException
        ("Requested time ... but the earliest data is at time ...")이 간헐적으로
        발생할 수 있음(발견 경위는 HISTORY.md §1 참고). spin_once를 직접
        반복 펌핑하며 짧게 재시도해 흡수함.
        """
        timeout_sec = self.mission_exec_cfg.get('tf_lookup_retry_sec', 1.0)
        deadline = time.time() + timeout_sec
        last_err = None
        while True:
            try:
                return self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            except Exception as e:
                last_err = e
                if time.time() >= deadline:
                    print(f"[!] TF lookup failed for map->base_link after "
                          f"{timeout_sec:.1f}s retry: {last_err}")
                    return None
                self.spin_executor.spin_once(timeout_sec=0.05)
                time.sleep(0.02)

    def _get_current_yaw_from_tf(self):
        trans = self._lookup_base_link_transform()
        if trans is None:
            return None
        return self._quaternion_to_yaw(trans.transform.rotation)

    def _get_current_pose_from_tf(self):
        """TF(map->base_link)에서 현재 (x, y, yaw)를 함께 읽어옴."""
        trans = self._lookup_base_link_transform()
        if trans is None:
            return None, None, None
        x = trans.transform.translation.x
        y = trans.transform.translation.y
        yaw = self._quaternion_to_yaw(trans.transform.rotation)
        return x, y, yaw


    def _rotate_in_place_to(self, target_pose, label='rotate'):
        """
        현재 위치에서 target_pose(위치)를 향하도록 회전함.

        target_pose.orientation을 그대로 쓰지 않음 - 그 값은 "target_pose에
        도착한 뒤 다음 지점을 향해야 할 방향"으로 저장된 값이라(translator.py의
        forward-looking 방식), 아직 target_pose에 도착 전인 지금 그 방향을 미리
        향하면 코너 직전 지점에서 코너를 건너뛰고 그 다음 방향을 미리 보게 됨.

        따라서 "현재 위치 -> target_pose 위치"를 atan2로 직접 계산해서 목표각으로 씀.
        """
        current_x, current_y, current_yaw = self._get_current_pose_from_tf()
        if current_yaw is None:
            return False

        dx = target_pose.pose.position.x - current_x
        dy = target_pose.pose.position.y - current_y
        if math.hypot(dx, dy) < 1e-3:
            print("  [Rotate] Target is at current position. Skipping spin.")
            return True

        target_yaw = math.atan2(dy, dx)
        delta_yaw = target_yaw - current_yaw
        delta_yaw = math.atan2(math.sin(delta_yaw), math.cos(delta_yaw))

        min_rotation_rad = math.radians(self.mission_exec_cfg.get('min_rotation_deg', 3.0))
        if abs(delta_yaw) < min_rotation_rad:
            print(f"  [Rotate] Already aligned (delta={math.degrees(delta_yaw):.1f}°). Skipping spin.")
            return True

        print(f"  [Rotate] current={math.degrees(current_yaw):.1f}°, "
            f"target={math.degrees(target_yaw):.1f}° (toward next goal), delta={math.degrees(delta_yaw):.1f}°")

        spin_time_allowance = self.mission_exec_cfg.get('spin_time_allowance_sec', 15.0)
        self.navigator.spin(spin_dist=delta_yaw, time_allowance=int(spin_time_allowance))

        stall_watcher = StallWatcher(f"{label} [rotate]", stall_threshold_sec=self._stall_threshold_sec())
        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self.navigator, timeout_sec=0.01)
            self.spin_executor.spin_once(timeout_sec=0.0)
            if self._check_amcl_jump():
                self.navigator.cancelTask()
                self._stall_events.extend(stall_watcher.finalize())
                return False
            feedback = self.navigator.getFeedback()
            stall_watcher.update(getattr(feedback, 'angular_distance_traveled', None) if feedback else None)
            time.sleep(0.05)
        self._stall_events.extend(stall_watcher.finalize())

        result = self.navigator.getResult()
        if result != TaskResult.SUCCEEDED:
            print(f"[!] Warning: Spin action did not succeed (result={result}).")
            return False
        return True

    def _stall_threshold_sec(self):
        return self.mission_exec_cfg.get('stall_log_threshold_sec', 5.0)

    # ------------------------------------------------------------------
    # 직선 주행 (nav2_msgs/action/NavigateToPose, coverage run의 끝점으로 1회 전송)
    # ------------------------------------------------------------------

    def _navigate_to_pose_blocking(self, pose, label='drive-to-pose'):
        self.navigator.goToPose(pose)

        stall_watcher = StallWatcher(f"{label} [drive]", stall_threshold_sec=self._stall_threshold_sec())
        last_debug_print_time = time.time()
        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self.navigator, timeout_sec=0.01)
            self.spin_executor.spin_once(timeout_sec=0.0)
            current_time = time.time()

            if self._check_amcl_jump():
                self.navigator.cancelTask()
                self._stall_events.extend(stall_watcher.finalize())
                return False

            feedback = self.navigator.getFeedback()
            remaining = getattr(feedback, 'distance_remaining', None) if feedback else None
            recoveries = getattr(feedback, 'number_of_recoveries', None) if feedback else None
            stall_watcher.update(remaining, recoveries=recoveries)

            if current_time - last_debug_print_time >= 1.0:
                if remaining is not None:
                    print(f"  ├─ Driving swath... distance remaining: {remaining:.2f}m")
                last_debug_print_time = current_time

            time.sleep(0.05)
        self._stall_events.extend(stall_watcher.finalize())

        result = self.navigator.getResult()
        if result != TaskResult.SUCCEEDED:
            print(f"[-] Swath drive ended without SUCCEEDED (result={result}).")
            return False
        return True

    def _set_angular_dist_threshold(self, mode):
        """
        FollowPath.angular_dist_threshold를 coverage/transit에 따라 실행 중에
        바꿈. coverage(정밀 측정 구간)는 yaml 기본값을 유지해 정확한 제자리
        회전을 쓰고, transit(코너/문지방 통과 구간)은 사실상 무제한에 가깝게
        풀어서 RotationShimController의 강제 제자리 회전 자체를 비활성화함 -
        좁은 공간에서 제자리 회전이 벽에 막혀 못 빠져나오는 문제의 대응임
        (waffle.yaml 값은 정적이라 재시작 없인 못 바꾸므로 set_parameters로
        실행 중에 전환함).
        """
        if mode == getattr(self, '_current_angular_mode', None):
            return True

        from rcl_interfaces.srv import SetParameters
        from rcl_interfaces.msg import Parameter as RclParameter, ParameterValue, ParameterType

        if self._controller_param_client is None:
            self._controller_param_client = self.create_client(
                SetParameters, '/controller_server/set_parameters'
            )

        coverage_threshold = self.mission_exec_cfg.get('coverage_angular_dist_threshold', 0.780)
        transit_threshold = self.mission_exec_cfg.get('transit_angular_dist_threshold', 3.140)
        value = coverage_threshold if mode == 'coverage' else transit_threshold

        if not self._controller_param_client.wait_for_service(timeout_sec=2.0):
            print("[!] Warning: /controller_server/set_parameters unavailable. "
                  "Threshold switch skipped - RotationShim will keep using the static yaml value.")
            return False

        param = RclParameter(
            name='FollowPath.angular_dist_threshold',
            value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=value)
        )
        request = SetParameters.Request(parameters=[param])
        future = self._controller_param_client.call_async(request)

        spin_start = time.time()
        while not future.done() and (time.time() - spin_start) < 2.0:
            self.spin_executor.spin_once(timeout_sec=0.1)

        ok = (future.done() and future.result() is not None
              and all(r.successful for r in future.result().results))
        if ok:
            self._current_angular_mode = mode
            print(f"  [Threshold] FollowPath.angular_dist_threshold -> {value:.3f} rad (mode={mode})")
        else:
            print(f"  [!] Warning: Failed to set angular_dist_threshold (mode={mode}).")
        return ok

    def _navigate_through_poses_blocking(self, seg_poses, label='drive-through-poses'):
        """
        seg_poses 전체를 NavigateThroughPoses(goThroughPoses)로 한 번에 전달함.
        goToPose(end_pose)만 보내면 중간 지점(특히 코너 꼭짓점)을 글로벌 플래너가
        반드시 지나가야 할 이유가 없어 코너를 넓게 잘라가며 지나가는 문제가
        생김. NavigateThroughPoses는 리스트의 모든 (x,y)를 반드시 통과해야
        하는 지점으로 취급하므로 코너 꼭짓점(seg_poses[0])을 실제로 스치듯
        지나가도록 강제할 수 있음.

        주의: 중간 지점들의 orientation은 글로벌 플래너가 강제하지 않음
        (위치만 통과 지점으로 취급됨). 최종 목표(seg_poses[-1])의 orientation만
        도착 시 정렬 대상이 됨.
        """
        self.navigator.goThroughPoses(seg_poses)

        stall_watcher = StallWatcher(f"{label} [drive]", stall_threshold_sec=self._stall_threshold_sec())
        last_debug_print_time = time.time()
        while not self.navigator.isTaskComplete():
            rclpy.spin_once(self.navigator, timeout_sec=0.01)
            self.spin_executor.spin_once(timeout_sec=0.0)
            current_time = time.time()

            if self._check_amcl_jump():
                self.navigator.cancelTask()
                self._stall_events.extend(stall_watcher.finalize())
                return False

            feedback = self.navigator.getFeedback()
            remaining = getattr(feedback, 'distance_remaining', None) if feedback else None
            n_left = getattr(feedback, 'number_of_poses_remaining', None) if feedback else None
            recoveries = getattr(feedback, 'number_of_recoveries', None) if feedback else None
            stall_watcher.update(remaining, recoveries=recoveries)

            if current_time - last_debug_print_time >= 1.0:
                # if remaining is not None:
                #     print(f"  ├─ Driving through {len(seg_poses)} points... "
                #         f"distance remaining: {remaining:.2f}m, poses left: {n_left}")
                last_debug_print_time = current_time

            time.sleep(0.05)
        self._stall_events.extend(stall_watcher.finalize())

        result = self.navigator.getResult()
        if result != TaskResult.SUCCEEDED:
            print(f"[-] Through-poses drive ended without SUCCEEDED (result={result}).")
            return False
        return True

    # ------------------------------------------------------------------
    # 캡처 시퀀스 보조 유틸 (settle 대기, surface_profiler 서비스 호출)
    # ------------------------------------------------------------------

    def _spin_sleep(self, duration_sec):
        """AMCL/네트워크 통신이 끊기지 않도록 spin을 유지하면서 duration_sec만큼 대기함."""
        end_time = time.time() + duration_sec
        while time.time() < end_time:
            self.spin_executor.spin_once(timeout_sec=0.05)
            if self.navigator is not None:
                rclpy.spin_once(self.navigator, timeout_sec=0.01)
            time.sleep(0.02)

    def _call_capture_service(self, client, label):
        """
        surface_profiler.py의 start/stop_waypoint_capture Trigger 서비스를 호출함.
        best-effort: 서버가 없거나 응답이 없어도 미션 자체를 막지 않고 경고만 남김.
        """
        service_name = client.srv_name
        if not client.wait_for_service(timeout_sec=2.0):
            print(f"[!] Warning: Service '{service_name}' not available. "
                  f"Is surface_profiler.py running on the notebook? Skipping {label}.")
            return False

        request = Trigger.Request()
        future = client.call_async(request)

        spin_start = time.time()
        while not future.done() and (time.time() - spin_start) < 5.0:
            self.spin_executor.spin_once(timeout_sec=0.1)

        if future.done() and future.result() is not None:
            response = future.result()
            print(f"[*] {label}: success={response.success}, message='{response.message}'")
            return response.success
        else:
            print(f"[!] Warning: No response from '{service_name}' within timeout.")
            return False

    # ------------------------------------------------------------------
    # surface_profiling(노트북) 측에 PCD 수집 종료 신호 전달
    # ------------------------------------------------------------------

    def _notify_surface_profiling_stop(self, success: bool, message: str = ""):
        """
        노트북에서 구동 중인 SurfaceProfiler에게 수집 종료를 알림.
        success=True면 정상 종료 서비스, False면 비정상(즉시 강제 종료) 서비스를 호출함.
        서버가 아직 떠 있지 않거나 응답이 없어도 미션 자체의 종료를 막지는 않음
        (호출은 best-effort로 처리하고, 결과만 로그로 남김).
        """
        client = self.stop_collection_success_client if success else self.stop_collection_abort_client
        service_name = client.srv_name

        if not client.wait_for_service(timeout_sec=3.0):
            print(f"[!] Warning: Service '{service_name}' not available. "
                  f"Is surface_profiler.py running on the notebook? Skipping notification.")
            return

        request = Trigger.Request()
        future = client.call_async(request)

        spin_start = time.time()
        while not future.done() and (time.time() - spin_start) < 5.0:
            self.spin_executor.spin_once(timeout_sec=0.1)

        if future.done() and future.result() is not None:
            response = future.result()
            print(f"[*] Notified '{service_name}': success={response.success}, message='{response.message}'")
        else:
            print(f"[!] Warning: No response from '{service_name}' within timeout.")

    # ------------------------------------------------------------------
    # 결과 저장
    # ------------------------------------------------------------------

    def _write_stall_report(self):
        """execute_mission() 동안 쌓인 self._stall_events(5초 이상 진행이 멈춘
        구간, StallWatcher 참고)만 따로 CSV 한 파일에 출력함 - 다른 로그와
        섞이지 않게 해서 grep/정렬만으로 어디서 얼마나/왜(recoveries 발동 여부)
        지연됐는지 바로 확인할 수 있게 하기 위함. 성공/실패/취소와 무관하게
        항상 호출됨."""
        if self._mission_start_wall_time is None:
            return

        log_dir = os.path.join(self.workspace_root, self.mission_exec_cfg.get('stall_log_dir', 'analytics/logs'))
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"stall_report_{int(self._mission_start_wall_time)}.csv")
            write_stall_report(
                self._stall_events, log_path, self._mission_start_wall_time,
                threshold_sec=self._stall_threshold_sec(),
            )
            print(f"[*] Stall report ({len(self._stall_events)} stall(s) >= "
                  f"{self._stall_threshold_sec():.1f}s) saved to: {log_path}")
        except Exception as e:
            print(f"[-] Failed to write stall report: {e}")
            traceback.print_exc()

    def _save_mission_results(self):
        # FollowWaypoints는 coverage 지점마다 별도의 goal로 나뉘어 순차 전송되므로,
        # navigator.getResult()는 "마지막으로 보낸 세그먼트" 하나의 결과만 반영함.
        # 미션 전체의 성공/실패는 execute_mission()에서 추적한 self.mission_succeeded를
        # 우선 참조하고, result는 로그 참고용으로만 사용함.
        result = self.navigator.getResult()
        overall_success = self.mission_succeeded

        # 주행 지연(stall) 리포트는 성공/실패/취소와 무관하게 항상 남김 -
        # "끝까지 완주는 했지만 중간중간 오래 멈췄던 구간"을 디버깅하는 것이
        # 목적이므로, 오히려 실패한 실행에서도 마지막 stall이 실패 원인일 수
        # 있어 더 중요함. 다른 로그와 섞이지 않도록 별도 파일 하나에만 씀.
        self._write_stall_report()

        if overall_success:
            print("[+] Mission Successfully Completed!")
            self._notify_surface_profiling_stop(success=True, message="Mission completed successfully.")

            # CSV 데이터 저장
            output_path_dir = os.path.join(self.workspace_root, self.mission_exec_cfg.get('output_path_dir', 'analytics/paths'))
            os.makedirs(output_path_dir, exist_ok=True)
            csv_filename = os.path.join(output_path_dir, f"robot_path_{int(time.time())}.csv")

            try:
                with open(csv_filename, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp", "x", "y"])
                    writer.writerows(self.path_history)
                print(f"[+] Successfully saved robot path history to '{csv_filename}'.")

                # 결과 시각화(PNG) 저장
                vis_dir = os.path.join(self.workspace_root, self.mission_exec_cfg.get('visualization_dir', 'visualization/mission_execution'))
                os.makedirs(vis_dir, exist_ok=True)
                img_out_path = os.path.join(vis_dir, f"robot_path_{int(time.time())}_plot.png")

                visualize_paths(csv_filename, self.final_path, img_out_path)

                risk_img_path = os.path.join(vis_dir, f"robot_path_{int(time.time())}_wall_risk.png")
                visualize_planned_wall_proximity(
                    self.final_path, self.map_yaml_path, risk_img_path,
                    robot_radius_m=self.env_cfg.get('robot_width', 0.15),
                )

                stall_img_path = os.path.join(vis_dir, f"robot_path_{int(time.time())}_stall.png")
                visualize_stall_points(csv_filename, stall_img_path)

            except Exception as e:
                print(f"[-] Failed to save outputs due to error: {e}")
                traceback.print_exc()

        elif result == TaskResult.CANCELED:
            print(f"\n[!] Mission was canceled! (last segment result={result})")
            self._notify_surface_profiling_stop(success=False, message="Mission was canceled.")
        else:
            print(f"\n[-] Mission failed! (last segment result={result})")
            self._notify_surface_profiling_stop(success=False, message="Mission failed.")

        # ROS 2 자원 안전 셧다운 (데드락 방지: subscription들은 execute_mission/_initialize_localization
        # 단계에서 이미 정리되었으므로 여기서는 executor/노드/컨텍스트 종료만 수행함)
        self.spin_executor.remove_node(self)
        self.destroy_node()
        if self.navigator is not None:
            self.navigator.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    # ------------------------------------------------------------------
    # 외부 실행 엔트리포인트
    # ------------------------------------------------------------------

    def run(self):
        self._setup_ros_environment()
        self._initialize_localization()
        self.execute_mission()
        self._save_mission_results()

def main(args=None):
    if args is None:
        args = sys.argv

    if not rclpy.ok():
        rclpy.init(args=args)

    mission_executor = MissionExecutor()
    mission_executor.run()

if __name__ == "__main__":
    main(args=sys.argv)