# launch/mission_execution.launch.py

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    """
    Jetson Orin Nano(turtlebot3 Waffle)에서 MissionExecutor(mission_executor.py) 실행

    전제 조건:
    1. turtleBot3 bringup이 이미 실행 중이어야 함.
    2. turtlebot3_navigation2의 navigation2.launch.py(map_server, amcl,
      lifecycle_manager, controller_server, planner_server, bt_navigator 등 포함)가
      이미 실행 중이어야 함.

    이 launch 파일은 그 위에서 mission_executor 노드 하나만 추가로 띄우는 역할만 함.
      (Nav2 스택 중복 실행 방지)

    Terminal Command:
        ros2 launch dae_coverage_floor_flatness mission_execution.launch.py is_sim:=true
        ros2 launch dae_coverage_floor_flatness mission_execution.launch.py is_sim:=false
    """

    is_sim_arg = DeclareLaunchArgument(
        'is_sim',
        default_value='false',
        description='true: Gazebo Simulation Mode, false: Real-world Mode'
    )

    is_sim = LaunchConfiguration('is_sim')

    mission_executor_node = Node(
        package='dae_coverage_floor_flatness',
        executable='mission_executor',
        name='mission_executor_node',
        output='screen',
        emulate_tty=True,
        parameters=[{
            'is_sim': is_sim,
            'use_sim_time': is_sim,
        }]
    )

    ld = LaunchDescription()
    ld.add_action(is_sim_arg)
    ld.add_action(mission_executor_node)

    return ld