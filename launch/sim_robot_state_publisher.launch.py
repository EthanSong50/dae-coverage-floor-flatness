#!/usr/bin/env python3
#
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Darby Lim

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'waffle')
    
    # 런치 인자 및 설정 정의
    namespace = LaunchConfiguration('namespace', default='')
    frame_prefix = LaunchConfiguration('frame_prefix', default='')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    # turtlebot3 패키지가 아닌 dae_coverage_floor_flatness/urdf 폴더 참조.
    pkg_my_dir = get_package_share_directory('dae_coverage_floor_flatness')
    urdf_file_path = os.path.join(
        pkg_my_dir,
        'urdf',
        f'turtlebot3_{TURTLEBOT3_MODEL}.urdf.xacro'
    )

    print(f'xacro_file_path : {urdf_file_path}')

    # xacro 파싱 (namespace 인자 전달)
    robot_desc = Command([
        'xacro ',
        urdf_file_path,
        ' namespace:=',
        PythonExpression(['"', namespace, '" + "/" if "', namespace, '" != "" else ""']),
    ])

    rsp_params = {'robot_description': ParameterValue(robot_desc, value_type=str)}

    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),
        DeclareLaunchArgument(
            'namespace',
            default_value='',
            description='Robot namespace'),
        DeclareLaunchArgument(
            'frame_prefix',
            default_value='',
            description='Frame prefix'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[
                rsp_params,
                {
                    'use_sim_time': use_sim_time,
                    'frame_prefix': PythonExpression(["'", frame_prefix, "/'"])
                }
            ],
        ),
    ])