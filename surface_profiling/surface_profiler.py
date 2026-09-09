# surface_profiling/surface_profiler.py

import os
import sys
import time
import math

import yaml
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np
import torch
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import tf_transformations
from std_srvs.srv import Trigger

try:
    from utils.floor_extractor import extract_floor_by_height
    from utils.heatmap_generator import generate_floor_heatmap
    from utils.pcd_io import export_pcd_to_csv
    from utils.stall_report_analyzer import analyze_and_visualize_stalls
    from utils.config_paths import (
        load_full_config as _load_full_config_shared,
        resolve_pointcloud_dir,
        resolve_visualization_dir,
        resolve_map_yaml_path,
    )
except ImportError:
    from surface_profiling.utils.floor_extractor import extract_floor_by_height
    from surface_profiling.utils.heatmap_generator import generate_floor_heatmap
    from surface_profiling.utils.pcd_io import export_pcd_to_csv
    from surface_profiling.utils.stall_report_analyzer import analyze_and_visualize_stalls
    from surface_profiling.utils.config_paths import (
        load_full_config as _load_full_config_shared,
        resolve_pointcloud_dir,
        resolve_visualization_dir,
        resolve_map_yaml_path,
    )


class SurfaceProfiler(Node):
    """
    3D라이다와 랜선으로 직결되어 pcd를 처리하는 노트북에서 구동되는 바닥 평탄도 측정 파이프라인 진입점 ROS 2 노드.

    1단계 (수집): Velodyne VLP-16 PointCloud2를 구독하며 TF('map' <- 'velodyne_link')를
                  통해 전역 좌표계로 정렬, 종료 신호를 받을 때까지 누적 후 PCD로 저장함.
                  TF는 같은 ROS2 도메인 내에서 Jetson(주행 시스템)이 publish하는 좌표 정보를
                  네트워크를 통해 직접 구독하는 구조이며,
                  Chrony 기반 시간 동기화 정밀도에 timestamp 매칭 신뢰도가 의존함.
    2단계 (바닥 추출): 높이(z) 기준으로 바닥면 후보 포인트만 필터링.
    3단계 (시각화): 필터링된 바닥면 데이터를 X-Y 격자 평균 높이로 변환해 히트맵 PNG 생성.
    4단계 (주행 지연 분석): 히트맵까지 다 만든 직후, mission_executor.py(Jetson)가
                  같은 미션에서 남긴 stall_report_<ts>.csv(utils/stall_logger.py)를
                  robot_path_<ts>.csv 궤적과 좌표로 대조해 분석 요약(analytics/logs)과
                  위치 시각화(visualization/mission_execution)를 자동 생성함
                  (utils/stall_report_analyzer.py). 실패해도 1~3단계 결과에는 영향 없음.

    종료 트리거: ENTER 입력 대신, mission_executor.py(Jetson)가 호출하는 두 개의
    std_srvs/Trigger 서비스로 종료를 제어함.
      - /surface_profiling/stop_collection_success : 미션 정상 완료. 수집을 멈추고
        2, 3단계까지 정상적으로 진행함.
      - /surface_profiling/stop_collection_abort : 미션 비정상 종료(타임아웃/실패/취소). 즉시 수집을 멈추되,
      그때까지 모은 포인트는 저장하고 2, 3단계도 동일하게 진행함. 이때 파일명에 '_aborted'라고 붙여 불완전한 데이터라는 뜻으로 저장함.

    지점별(Waypoint) 캡처: coverage 웨이포인트에서 로봇이 정지하는 동안만 정밀 데이터를
    모으기 위해, mission_executor.py가 호출하는 시작/종료 한 쌍의 std_srvs/Trigger
    서비스로 캡처 구간(on/off)을 제어함. 정착 대기(settling)와 실제 캡처 시간(active
    capture) 관리는 전부 mission_executor(주행 측)가 전담하며, 이 노드는 신호에만 반응함
    (주행과 측정의 기능적 분리 원칙 유지).
      - /surface_profiling/start_waypoint_capture : 이 시점부터 들어오는 포인트를
        지점 전용 버퍼(self.current_waypoint_points)에 적재하기 시작함.
      - /surface_profiling/stop_waypoint_capture : 적재를 멈추고, 모인 포인트를
        해당 지점 전용 PCD 파일로 저장한 뒤, 최종 통합 히트맵을 위해 마스터 버퍼
        (self.all_points)에도 병합함.

    params.yaml의 surface_profiling.only_capture_at_waypoints (기본값 True)로 동작을
    전환할 수 있음: True면 정지-캡처 구간의 포인트만 최종 결과에 반영하고(이동 중
    포인트는 모션 블러 우려로 버림), False면 기존처럼 전 구간을 연속 수집하되
    지점별 캡처 파일은 부가적으로만 별도 저장함.

    부가 기능: 필요 시 PCD를 CSV로 변환하는 보조 유틸(utils.pcd_io)을 제공.
    
    ROS 2 Node 내장 executor와의 이름 충돌 회피:
    - self.spin_executor
    - Node.executor는 read-only property로 존재하므로 직접 할당 불가
    - Python Descriptor Protocol에 의해 할당 시도가 무시되고 None이 됨
    """

    PCD_FILENAME_TEMPLATE = "combined_{ts}{suffix}.pcd"
    PCD_FILTERED_FILENAME_TEMPLATE = "combined_filtered_{ts}{suffix}.pcd"
    CSV_FILENAME_TEMPLATE = "combined_{ts}{suffix}.csv"
    HEATMAP_FILENAME_TEMPLATE = "floor_heatmap_{ts}{suffix}.png"
    WAYPOINT_PCD_FILENAME_TEMPLATE = "waypoint_{idx:04d}_{ts}.pcd"

    def __init__(self):
        super().__init__('surface_profiler_node')

        print("\n=======================================================")
        print("[*] Surface Profiler (3D LiDAR Floor Flatness Measurement)")
        print("=======================================================\n")

        # self.spin_executor
        # self 전용 SingleThreadedExecutor. 이 노드는 ROS 통신을 전부 직접 처리하는
        # 단일 노드이므로(mission_executor.py처럼 별도 navigator Node가 없음),
        # executor 분리 없이 self 하나만 등록.
        self.spin_executor = SingleThreadedExecutor()
        self.spin_executor.add_node(self)

        # 상태 변수 초기화
        self.config = None
        self.global_cfg = None
        self.profiling_cfg = None
        self.mission_exec_cfg = None
        self.workspace_root = None

        self.pointcloud_dir = None
        self.visualization_dir = None
        self.map_yaml_path = None

        # 수집 종료 제어 상태
        self.stop_requested = False
        self.is_aborted = False
        self.collection_start_ts = None  # 'YYYY-MM-DD_HH-MM-SS' 형식, 파일명 공용 타임스탬프

        # 포인트클라우드 누적 버퍼 (최종 히트맵용 마스터 버퍼)
        self.all_points = []

        # 지점별(Waypoint) 캡처 관련 상태.
        # capture_active=True인 동안에만 _pc_callback이 포인트를
        # self.current_waypoint_points에 적재함(정지 상태에서 모은 깨끗한
        # 데이터만 반영하기 위함). only_capture_at_waypoints 실제 값은
        # _load_config()에서 profiling_cfg를 읽은 뒤 갱신됨.
        self.capture_active = False
        self.waypoint_capture_counter = 0
        self.current_waypoint_points = []
        self.waypoint_pointcloud_dir = None  # _resolve_directories()에서 설정
        self.only_capture_at_waypoints = True

        # TF 저역통과 필터 상태. 연속된 두 PointCloud2 프레임 사이의 순간
        # 선속도/각속도가 임계치(params.yaml)를 넘으면 해당 프레임을 통째로
        # 버림. 실제 값(enable/threshold)은 _load_config()에서 갱신됨.
        self.enable_tf_lowpass_filter = True
        self.tf_lowpass_max_linear_vel = 0.3       # m/s
        self.tf_lowpass_max_angular_vel_deg = 20.0  # deg/s
        self.last_tf_translation = None
        self.last_tf_yaw = None
        self.last_tf_stamp = None

        # 1. ROS 2 파라미터로 시뮬레이션 모드 여부 확보 (launch argument로 주입됨)
        self._resolve_run_mode()

        # 2. params.yaml 설정 파일 파싱
        self._load_config()

        # 3. 외부 저장소 디렉토리 확보
        self._resolve_directories()

        # 4. TF, PointCloud2 구독, 종료/캡처 트리거 서비스 서버 설정
        self._setup_tf()
        self._setup_pointcloud_subscription()
        self._setup_stop_services()
        self._setup_waypoint_capture_services()

    # ------------------------------------------------------------------
    # 초기화 단계 헬퍼
    # ------------------------------------------------------------------

    def _resolve_run_mode(self):
        """
        'is_sim' ROS 2 파라미터를 선언하고 읽음. launch 파일이 주입하지 않으면
        기본값 False(Real-world)로 동작.
        """
        self.declare_parameter('is_sim', False)
        self.is_sim = self.get_parameter('is_sim').get_parameter_value().bool_value

        if self.is_sim:
            self.get_logger().info("Running Simulation(Gazebo) Mode.")
        else:
            self.get_logger().info("Running Real-world Mode.")

        # use_sim_time 강제 동기화 (launch 파일이 누락했을 경우 안전장치)
        use_sim_time_param = self.get_parameter_or(
            'use_sim_time', Parameter('use_sim_time', Parameter.Type.BOOL, self.is_sim)
        )
        if use_sim_time_param.value != self.is_sim:
            self.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, self.is_sim)])

    def _load_config(self):
        # [변경] 경로 해석 로직을 utils.config_paths.load_full_config()로 일원화.
        # reprocess_pcd.py는 여전히 load_config()(surface_profiling 섹션만)를 쓰므로
        # 서로 다르게 해석하는 문제는 재발하지 않음. 여기서 load_full_config를
        # 쓰는 이유는 Stage 4(stall 분석)가 mission_execution 섹션(stall_log_dir/
        # output_path_dir 등)도 읽어야 하기 때문 - profiling_cfg는 그대로 전체
        # config에서 동일하게 잘라내 쓰므로 기존 동작과 차이 없음.
        self.workspace_root, self._full_config = _load_full_config_shared()
        self.profiling_cfg = self._full_config.get('surface_profiling', {})
        self.mission_exec_cfg = self._full_config.get('mission_execution', {})
        self.only_capture_at_waypoints = self.profiling_cfg.get('only_capture_at_waypoints', True)
        self.get_logger().info(
            f"only_capture_at_waypoints = {self.only_capture_at_waypoints} "
            f"({'정지-캡처 구간만 반영' if self.only_capture_at_waypoints else '연속 수집 + 지점별 부가 저장'})"
        )

        self.enable_tf_lowpass_filter = self.profiling_cfg.get('enable_tf_lowpass_filter', True)
        self.tf_lowpass_max_linear_vel = self.profiling_cfg.get('tf_lowpass_max_linear_vel', 0.3)
        self.tf_lowpass_max_angular_vel_deg = self.profiling_cfg.get('tf_lowpass_max_angular_vel_deg', 20.0)
        self.get_logger().info(
            f"TF Low-pass Filter: enabled={self.enable_tf_lowpass_filter}, "
            f"max_linear_vel={self.tf_lowpass_max_linear_vel}m/s, "
            f"max_angular_vel={self.tf_lowpass_max_angular_vel_deg}deg/s"
        )

    def _resolve_directories(self):
        # [변경] 경로 조합 로직을 utils.config_paths로 일원화.
        # 예전에는 여기서 map_yaml_dir을 완전한 파일 경로로 조합해뒀는데,
        # _run_visualization_stage()가 이를 쓰지 않고 profiling_cfg에서
        # 원본 값("maps/grid", 폴더명만)을 다시 읽어버려서 서로 어긋나는
        # 버그가 있었음(HISTORY.md 참고). resolve_map_yaml_path()가 조합한
        # 값을 self.map_yaml_path에 저장해두고, 아래 단계들이 전부 이
        # 하나의 값만 참조하도록 통일함.
        self.pointcloud_dir = resolve_pointcloud_dir(self.workspace_root, self.profiling_cfg)
        self.visualization_dir = resolve_visualization_dir(self.workspace_root, self.profiling_cfg)
        self.map_yaml_path = resolve_map_yaml_path(self.workspace_root, self.profiling_cfg)

        # 지점별(Waypoint) 캡처 PCD 저장용 서브폴더. 추후 지점 단위 재처리가
        # 가능하도록 combined PCD와 분리해서 보관함.
        self.waypoint_pointcloud_dir = os.path.join(self.pointcloud_dir, 'waypoints')
        os.makedirs(self.waypoint_pointcloud_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # TF / PointCloud2 구독 설정 
    # ------------------------------------------------------------------

    def _setup_tf(self):
        self.target_frame = self.profiling_cfg.get('target_frame', 'map')
        self.source_frame = self.profiling_cfg.get('source_frame', 'velodyne_link')
        self.tf_timeout_sec = self.profiling_cfg.get('tf_timeout_sec', 0.1)

        self.tf_buffer = Buffer()
        # [변경] spin_thread=True를 쓰지 않는다(기본값 False).
        # 예전 코드(lookup_transform을 콜백 안에서 timeout까지 블로킹 대기)에서는
        # tf 구독을 별도 스레드로 분리하는 게 필수였지만, 지금은
        # wait_for_transform_async(코루틴, await로 양보)로 바뀌어서 콜백이
        # 스레드를 블로킹하지 않음. 따라서 tf 구독도 self.spin_executor
        # 하나로 충분히 처리되고, 별도 스레드/executor가 없으니 노드 종료
        # 시점에 "아직 살아있는 백그라운드 스레드 vs rclpy.shutdown()" 같은
        # 레이스 컨디션(ExternalShutdownException)도 원천적으로 사라짐.
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def _setup_pointcloud_subscription(self):
        self.voxel_size = self.profiling_cfg.get('voxel_size', 0.01)

        topic_name = (
            self.profiling_cfg.get('pcd_topic_sim', '/velodyne_points')
            if self.is_sim
            else self.profiling_cfg.get('pcd_topic_real', '/velodyne_points')
        )
        self.get_logger().info(f"Execution Mode: {'Simulation' if self.is_sim else 'Real Hardware'}")
        self.get_logger().info(f"Subscribing to topic: {topic_name}")

        # GPU 사용 가능 여부 확인
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        # [TF-PointCloud 동기화 구조]
        # map->odom(AMCL, 실측 약 5Hz)이 velodyne_points(10Hz)보다 느려서, 단순
        # lookup_transform(exact_stamp, timeout)만으로는 tf2가 그 시각을 보간할
        # 다음 샘플이 아직 안 왔다는 이유로 대부분 실패한다(timeout을 늘려도 무의미).
        #
        # tf2_ros.MessageFilter는 C++ 전용 API라 rclpy(Python)에는 존재하지 않으므로,
        # 여기서는 Buffer.wait_for_transform_async(코루틴)를 이용해 동일한 효과를
        # 직접 구현함: 메시지를 즉시 처리하지 않고, 해당 stamp를 커버하는 TF가
        # 실제로 도착할 때까지 코루틴으로 기다렸다가 준비되면 콜백을 실행함.
        # 대기 중에도 콜백/실행기는 블로킹되지 않고(await로 양보), 그동안 새로 들어오는
        # 메시지도 계속 큐에 쌓임. _pending_queue_maxlen은 TF 최대 주기(약 0.5s)
        # 동안 들어오는 pointcloud 개수(~5개)보다 넉넉하게 잡음.
        # 대기 중인(TF 준비를 기다리는) 메시지에 대한 참조 보관용.
        # 자동 만료(maxlen) 없이 완료 시점에만 명시적으로 제거한다 —
        # deque(maxlen=N)을 쓰면 아직 처리 중인 태스크의 메시지가
        # 강제로 밀려나 예기치 않게 끊길 수 있기 때문.
        self._pending_pc_msgs = set()

        self.pc_subscription = self.create_subscription(
            PointCloud2,
            topic_name,
            self._pc_enqueue_callback,
            10
        )

    def _pc_enqueue_callback(self, msg: PointCloud2):
        """PointCloud2 메시지를 즉시 처리하지 않고, TF 준비를 기다리는 비동기 태스크로 등록만 함."""
        if self.stop_requested:
            return
        self._pending_pc_msgs.add(id(msg))
        # 코루틴을 태스크로 등록 -> executor가 spin하는 동안 백그라운드에서 진행됨.
        self.spin_executor.create_task(self._wait_and_process(msg))

    async def _wait_and_process(self, msg: PointCloud2):
        """해당 msg의 stamp를 커버하는 TF가 준비될 때까지 기다린 뒤 처리함."""
        try:
            await self.tf_buffer.wait_for_transform_async(
                self.target_frame,
                self.source_frame,
                msg.header.stamp,
            )
        except Exception as e:
            self.get_logger().warn(f"TF 대기 실패 원인: {e}")
            return
        finally:
            self._pending_pc_msgs.discard(id(msg))

        self._pc_callback(msg)

    def _tf_passes_lowpass_filter(self, trans: TransformStamped, msg_stamp):
        """
        연속된 두 PointCloud2 프레임 사이의 TF(target_frame->source_frame) 순간
        선속도/각속도를 계산해서, params.yaml의 임계치(tf_lowpass_max_linear_vel,
        tf_lowpass_max_angular_vel_deg)를 넘으면 False(버림)를 반환함.

        기준값(last_tf_*)은 "통과한" 프레임에서만 갱신함. 튄 프레임을 다음
        비교의 새 기준으로 삼아버리면, 그 다음 정상 프레임까지 연쇄적으로 오탐
        처리될 수 있기 때문임.
        """
        if not self.enable_tf_lowpass_filter:
            return True

        curr_t = np.array([
            trans.transform.translation.x,
            trans.transform.translation.y,
            trans.transform.translation.z,
        ])
        curr_q = trans.transform.rotation
        curr_yaw = 2.0 * math.atan2(curr_q.z, curr_q.w)
        curr_stamp = msg_stamp.sec + msg_stamp.nanosec * 1e-9

        if self.last_tf_translation is None:
            self.last_tf_translation = curr_t
            self.last_tf_yaw = curr_yaw
            self.last_tf_stamp = curr_stamp
            return True

        dt = curr_stamp - self.last_tf_stamp
        if dt <= 1e-4:
            # 타임스탬프가 동일하거나 역행함(비교 불가). 기준은 갱신하지 않고 일단 통과.
            return True

        linear_vel = np.linalg.norm(curr_t - self.last_tf_translation) / dt

        dyaw = curr_yaw - self.last_tf_yaw
        dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))  # -pi ~ +pi 정규화
        angular_vel_deg = abs(math.degrees(dyaw)) / dt

        passes = (linear_vel <= self.tf_lowpass_max_linear_vel and
                  angular_vel_deg <= self.tf_lowpass_max_angular_vel_deg)

        if passes:
            self.last_tf_translation = curr_t
            self.last_tf_yaw = curr_yaw
            self.last_tf_stamp = curr_stamp
        else:
            self.get_logger().warn(
                f"[TF Low-pass] Frame rejected: linear_vel={linear_vel:.2f}m/s "
                f"(limit {self.tf_lowpass_max_linear_vel}), "
                f"angular_vel={angular_vel_deg:.1f}deg/s "
                f"(limit {self.tf_lowpass_max_angular_vel_deg})."
            )

        return passes

    def _pc_callback(self, msg: PointCloud2):
        """_wait_and_process에서 TF 준비가 확인된 뒤에만 호출됨."""
        if self.stop_requested:
            return

        try:
            trans = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                msg.header.stamp,
            )
        except Exception as e:
            # wait_for_transform_async 통과 직후라 정상적으로는 거의 발생하지 않지만,
            # 버퍼 캐시 만료(오래된 TF가 밀려난 경우) 등 극히 드문 경합에 대비한 안전망.
            self.get_logger().warn(f"TF 변환 실패 원인: {e}")
            return

        # TF 저역통과 필터: 직전 프레임 대비 순간 선속도/각속도가 임계치를 넘으면
        # (AMCL 점프, 회전 중 잔여 프레임 등) 이 프레임 전체를 버림. 새 주행
        # 설계(mission_executor.py)가 회전 중엔 애초에 캡처를 켜지 않지만, 이건
        # 그래도 남을 수 있는 잔여 오차에 대한 방어선(defense in depth)임.
        if not self._tf_passes_lowpass_filter(trans, msg.header.stamp):
            return

        # PointCloud2 → numpy array 변환
        raw_points = pc2.read_points(msg, skip_nans=True, field_names=('x', 'y', 'z'))
        points_np = np.array([(p[0], p[1], p[2]) for p in raw_points], dtype=np.float32)
        if len(points_np) == 0:
            return

        # 행렬 연산을 위해 GPU(또는 CPU)로 전송
        points_t = torch.tensor(points_np, device=self.device)
        trans_mat_t = torch.tensor(self._transform_to_matrix(trans), device=self.device, dtype=torch.float32)

        # Homogeneous transformation
        ones = torch.ones((points_t.shape[0], 1), device=self.device)
        points_homo = torch.cat([points_t, ones], dim=1)

        # (4, 4) @ (4, N) 연산 후 전치
        transformed_t = (trans_mat_t @ points_homo.T).T

        # 결과 저장 (최종 저장 시에만 CPU로 복사)
        transformed_np = transformed_t[:, :3].cpu().numpy()

        if self.only_capture_at_waypoints:
            # 정지-캡처 구간(capture_active=True)의 포인트만 적재함.
            # 이동(transit) 중 수집된 포인트는 모션 블러/타임스탬프 오차 우려로 버림.
            if self.capture_active:
                self.current_waypoint_points.append(transformed_np)
        else:
            # 기존 동작(전 구간 연속 수집) 유지. 캡처 구간 포인트는 부가적으로
            # 지점별 버퍼에도 동시에 적재해서 별도 PCD로도 저장할 수 있게 함.
            self.all_points.append(transformed_np)
            if self.capture_active:
                self.current_waypoint_points.append(transformed_np)

    def _transform_to_matrix(self, trans: TransformStamped):
        t = [trans.transform.translation.x,
             trans.transform.translation.y,
             trans.transform.translation.z]
        q = [trans.transform.rotation.x,
             trans.transform.rotation.y,
             trans.transform.rotation.z,
             trans.transform.rotation.w]
        mat = tf_transformations.quaternion_matrix(q)
        mat[:3, 3] = t
        return mat

    # ------------------------------------------------------------------
    # 종료 트리거 서비스 서버
    # ------------------------------------------------------------------

    def _setup_stop_services(self):
        self.stop_success_srv = self.create_service(
            Trigger, '/surface_profiling/stop_collection_success', self._handle_stop_success
        )
        self.stop_abort_srv = self.create_service(
            Trigger, '/surface_profiling/stop_collection_abort', self._handle_stop_abort
        )
        self.get_logger().info(
            "Stop-trigger services ready: "
            "/surface_profiling/stop_collection_success, /surface_profiling/stop_collection_abort"
        )

    def _handle_stop_success(self, request, response):
        print("[*] Received STOP signal (mission succeeded). Stopping collection...")
        self.is_aborted = False
        self.stop_requested = True
        response.success = True
        response.message = "Collection stopped (mission succeeded)."
        return response

    def _handle_stop_abort(self, request, response):
        print("[!] Received ABORT signal (mission failed/canceled/timed out). "
              "Stopping collection immediately. Collected points so far will still be saved.")
        self.is_aborted = True
        self.stop_requested = True
        response.success = True
        response.message = "Collection aborted; partial data will be saved."
        return response

    # ------------------------------------------------------------------
    # 지점별(Waypoint) 캡처 시작/종료 트리거 서비스
    # ------------------------------------------------------------------

    def _setup_waypoint_capture_services(self):
        self.start_capture_srv = self.create_service(
            Trigger, '/surface_profiling/start_waypoint_capture', self._handle_start_waypoint_capture
        )
        self.stop_capture_srv = self.create_service(
            Trigger, '/surface_profiling/stop_waypoint_capture', self._handle_stop_waypoint_capture
        )
        self.get_logger().info(
            "Waypoint capture services ready: "
            "/surface_profiling/start_waypoint_capture, /surface_profiling/stop_waypoint_capture"
        )

    def _handle_start_waypoint_capture(self, request, response):
        # 방어 코드: start/stop은 항상 쌍으로 불리는 게 프로토콜상 전제지만,
        # 혹시 stop 없이 start가 재호출되면 미저장 포인트를 버리지 않고
        # 먼저 flush함(HISTORY.md §7 참고).
        if self.capture_active and self.current_waypoint_points:
            print(f"[!] Warning: start_waypoint_capture re-invoked while capture "
                  f"#{self.waypoint_capture_counter} was still active - flushing pending buffer first.")
            self._handle_stop_waypoint_capture(request, Trigger.Response())

        # 새 캡처 구간을 위해 지점 전용 버퍼를 비우고 적재를 켬.
        self.current_waypoint_points = []
        self.capture_active = True
        self.waypoint_capture_counter += 1

        print(f"[*] [Waypoint #{self.waypoint_capture_counter}] Capture window OPENED.")
        response.success = True
        response.message = f"Capture started (waypoint_id={self.waypoint_capture_counter})"
        return response

    def _handle_stop_waypoint_capture(self, request, response):
        self.capture_active = False
        captured = self.current_waypoint_points
        self.current_waypoint_points = []

        point_count = sum(arr.shape[0] for arr in captured) if captured else 0
        print(f"[*] [Waypoint #{self.waypoint_capture_counter}] Capture window CLOSED. "
              f"Points captured: {point_count}")

        if point_count == 0:
            print(f"[!] Warning: Waypoint #{self.waypoint_capture_counter} captured 0 points "
                  f"(TF/토픽 타이밍 문제이거나 정지 시간이 너무 짧을 수 있음).")
            response.success = True
            response.message = f"Capture stopped (waypoint_id={self.waypoint_capture_counter}, points=0)"
            return response

        waypoint_points_np = np.vstack(captured)

        # 1. 지점별 개별 PCD로 저장 (추후 지점 단위 재처리/디버깅용)
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        waypoint_pcd = o3d.geometry.PointCloud()
        waypoint_pcd.points = o3d.utility.Vector3dVector(waypoint_points_np)
        filename = self.WAYPOINT_PCD_FILENAME_TEMPLATE.format(
            idx=self.waypoint_capture_counter, ts=ts
        )
        filepath = os.path.join(self.waypoint_pointcloud_dir, filename)
        o3d.io.write_point_cloud(filepath, waypoint_pcd)
        print(f"[+] Saved per-waypoint PCD: {filepath}")

        # 2. 최종 통합 히트맵을 위해 마스터 버퍼에도 병합.
        #    only_capture_at_waypoints=False 모드에서는 _pc_callback이 이미
        #    all_points에도 동시 적재했으므로, 여기서 다시 넣으면 중복이라 생략함.
        if self.only_capture_at_waypoints:
            self.all_points.append(waypoint_points_np)

        response.success = True
        response.message = f"Capture stopped (waypoint_id={self.waypoint_capture_counter}, points={point_count})"
        return response

    # ------------------------------------------------------------------
    # 1단계: 수집 (메인 spin 루프)
    # ------------------------------------------------------------------

    def _run_collection_stage(self):
        print("\n[Stage 1/4] Point Cloud Collection")
        print("[*] Waiting for stop signal from mission_executor.py (Jetson)...")

        self.collection_start_ts = time.strftime("%Y-%m-%d_%H-%M-%S")

        try:
            while rclpy.ok() and not self.stop_requested:
                self.spin_executor.spin_once(timeout_sec=0.1)

            # STOP 신호 직후에도 TF 도착을 기다리던 프레임들(_pending_pc_msgs)이
            # 아직 몇 개 남아있을 수 있으므로, 짧게 마저 처리되도록 유예 시간을 줌.
            # (최대 tf_timeout_sec * 2 정도, 그 이상은 어차피 stop_requested로
            #  새 콜백 등록이 막혀 있으니 무한 대기로 이어지지 않음.)
            grace_deadline = time.time() + max(self.tf_timeout_sec * 2, 1.0)
            while self._pending_pc_msgs and time.time() < grace_deadline:
                self.spin_executor.spin_once(timeout_sec=0.05)
        except KeyboardInterrupt:
            print("[!] KeyboardInterrupt received. Stopping collection...")
            self.is_aborted = True

        suffix = "_aborted" if self.is_aborted else ""
        pcd_filename = self.PCD_FILENAME_TEMPLATE.format(ts=self.collection_start_ts, suffix=suffix)
        pcd_path = self._save_combined_pcd(pcd_filename=pcd_filename)

        if pcd_path is None:
            print("[!] CRITICAL ERROR: No points were collected. Aborting pipeline.")
            self._shutdown()
            sys.exit(1)

        return pcd_path

    def _save_combined_pcd(self, pcd_filename):
        """누적된 포인트들을 다운샘플링 후 단일 PCD 파일로 저장하고 경로를 반환함."""
        if not self.all_points:
            print("[-] No points collected.")
            return None

        print("[*] Stacking all frames...")
        all_points_np = np.vstack(self.all_points)
        print(f"[*] Total points collected: {all_points_np.shape[0]}")

        combined_pcd = o3d.geometry.PointCloud()
        combined_pcd.points = o3d.utility.Vector3dVector(all_points_np)

        print(f"[*] Downsampling PCD with voxel size: {self.voxel_size}m...")
        downsampled_pcd = combined_pcd.voxel_down_sample(voxel_size=self.voxel_size)

        pcd_file = os.path.join(self.pointcloud_dir, pcd_filename)
        o3d.io.write_point_cloud(pcd_file, downsampled_pcd)
        print(f"[+] Saved combined PCD: {pcd_file}")
        return pcd_file

    # ------------------------------------------------------------------
    # 2단계: 바닥 추출
    # ------------------------------------------------------------------

    def _run_floor_extraction_stage(self, pcd_path):
        print("\n[Stage 2/4] Floor Extraction (Height-based Filtering)")

        # z_min/z_max: 바닥으로 간주할 높이 밴드. params.yaml에 없으면
        # 바닥 요철이 보통 수 cm 이내인 것을 감안한 기본값 사용.
        z_min = self.profiling_cfg.get('z_min', -0.005)
        z_max = self.profiling_cfg.get('z_max', 0.035)
        suffix = "_aborted" if self.is_aborted else ""
        filtered_filename = self.PCD_FILTERED_FILENAME_TEMPLATE.format(ts=self.collection_start_ts, suffix=suffix)
        filtered_path = os.path.join(self.pointcloud_dir, filtered_filename)

        extract_floor_by_height(pcd_path, filtered_path, z_min=z_min, z_max=z_max)
        return filtered_path

    # ------------------------------------------------------------------
    # 3단계: 히트맵 시각화
    # ------------------------------------------------------------------

    def _run_visualization_stage(self, filtered_pcd_path):
        print("\n[Stage 3/4] Floor Flatness Heatmap Generation")

        grid_size = self.profiling_cfg.get('grid_size', 0.02)
        # 색상 스케일의 절대 범위. floor_extractor의 z_min/z_max와
        # 동일한 값을 쓰는 것을 권장 (필터링 범위 = 색상 표현 범위).
        z_min = self.profiling_cfg.get('z_min', -0.005)
        z_max = self.profiling_cfg.get('z_max', 0.035)
        # [수정] profiling_cfg.get('map_yaml_dir', ...)로 다시 읽으면 폴더명만
        # 있는 미완성 값("maps/grid")이 그대로 들어가 파일을 못 찾는 버그가
        # 있었음. _resolve_directories()가 이미 완전한 파일 경로로 조합해둔
        # self.map_yaml_path 하나만 참조하도록 통일.
        map_yaml_path = self.map_yaml_path
        suffix = "_aborted" if self.is_aborted else ""
        img_filename = self.HEATMAP_FILENAME_TEMPLATE.format(ts=self.collection_start_ts, suffix=suffix)
        img_out_path = os.path.join(self.visualization_dir, img_filename)

        generate_floor_heatmap(
            filtered_pcd_path,
            img_out_path,
            grid_size=grid_size,
            z_min=z_min,
            z_max=z_max,
            map_yaml_dir=map_yaml_path,
        )
        return img_out_path

    # ------------------------------------------------------------------
    # PCD -> CSV 변환
    # ------------------------------------------------------------------

    def export_csv(self, pcd_path=None):
        """PCD를 CSV로 변환하는 기능. run()의 파이프라인 마지막 단계로 항상 호출되지만,
        필요 시 외부에서 단독으로도 호출할 수 있음."""
        suffix = "_aborted" if self.is_aborted else ""
        if pcd_path is None:
            pcd_filename = self.PCD_FILENAME_TEMPLATE.format(ts=self.collection_start_ts, suffix=suffix)
            pcd_path = os.path.join(self.pointcloud_dir, pcd_filename)
        csv_filename = self.CSV_FILENAME_TEMPLATE.format(ts=self.collection_start_ts, suffix=suffix)
        csv_path = os.path.join(self.pointcloud_dir, csv_filename)
        return export_pcd_to_csv(pcd_path, csv_path)

    # ------------------------------------------------------------------
    # 자원 정리
    # ------------------------------------------------------------------

    def _shutdown(self):
        if hasattr(self, 'pc_subscription'):
            self.destroy_subscription(self.pc_subscription)
        if hasattr(self, 'tf_listener'):
            # spin_thread=False(기본값)이므로 별도 백그라운드 스레드가 없음.
            # tf 구독 정리만 하면 되고, join으로 기다려야 할 스레드가 없어
            # 이전에 있었던 ExternalShutdownException 레이스 컨디션도 없음.
            self.tf_listener.unregister()
        self.spin_executor.remove_node(self)
        self.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    # ------------------------------------------------------------------
    # 외부 실행 엔트리포인트
    # ------------------------------------------------------------------

    def run(self):
        pcd_path = self._run_collection_stage()
        filtered_pcd_path = self._run_floor_extraction_stage(pcd_path)
        img_out_path = self._run_visualization_stage(filtered_pcd_path)
        csv_path = self.export_csv(pcd_path=pcd_path)
        analyze_and_visualize_stalls(self.workspace_root, self.mission_exec_cfg)

        print("\n=======================================================")
        if self.is_aborted:
            print("[!] Surface Profiling Pipeline Completed (ABORTED - partial data).")
        else:
            print("[+] Surface Profiling Pipeline Completed Successfully.")
        print(f"    -> Raw PCD: {pcd_path}")
        print(f"    -> Filtered PCD: {filtered_pcd_path}")
        print(f"    -> Heatmap Image: {img_out_path}")
        print(f"    -> CSV Export: {csv_path}")
        print("=======================================================\n")

        self._shutdown()

def main(args=None):
    if args is None:
        args = sys.argv

    if not rclpy.ok():
        rclpy.init(args=args)

    profiler = SurfaceProfiler()
    profiler.run()


if __name__ == "__main__":
    main(args=sys.argv)
