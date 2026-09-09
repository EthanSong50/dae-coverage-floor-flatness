# launch/surface_profiling.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """
    3D라이다 데이터를 수신 중인 기기에서 SurfaceProfiler(surface_profiler.py)를 실행함.

    전제 조건:
    - 기기가 Jetson과 같은 ROS 2 도메인에 있어야 하며,
      Jetson이 publish하는 /tf, /tf_static을 네트워크로 수신할 수 있어야 함.
    - Velodyne VLP-16 드라이버가 노트북에 연결(랜선으로 직결 등)되어 PointCloud2를
      퍼블리시하고 있어야 함(velodyne_driver 등, 이 launch 파일이 띄우지 않음).
    - Chrony 시간 동기화가 Jetson과 노트북 사이에 맞춰져 있어야 TF lookup의
      timestamp 매칭이 정확함.

    이 노드는 시작과 동시에 /surface_profiling/stop_collection_success,
    /surface_profiling/stop_collection_abort 두 서비스를 열고 대기하므로,
    (Jetson Orin Nano에서의) mission_executor.py보다 먼저 실행되어
    있어야 함.

    Terminal Command:
        ros2 launch dae_coverage_floor_flatness surface_profiling.launch.py is_sim:=true
        ros2 launch dae_coverage_floor_flatness surface_profiling.launch.py is_sim:=false
    """

    is_sim_arg = DeclareLaunchArgument(
        'is_sim',
        default_value='false',
        description='true면 Gazebo 시뮬레이션 모드, false면 실제 로봇(Real-world) 모드로 동작'
    )

    is_sim = LaunchConfiguration('is_sim')

    surface_profiler_node = Node(
        package='dae_coverage_floor_flatness',
        executable='surface_profiler',
        name='surface_profiler_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'is_sim': is_sim,
            'use_sim_time': is_sim,
        }]
    )

    ld = LaunchDescription()
    ld.add_action(is_sim_arg)
    ld.add_action(surface_profiler_node)

    return ld