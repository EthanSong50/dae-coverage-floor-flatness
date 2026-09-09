# mission_execution/utils/boundary_repass.py

"""
"지나쳐야 채워진다" 원칙(mission_executor.py 참고)은 로봇이 한 지점을 접근 +
통과-후-멀어짐, 양방향으로 지나쳐야 라이다 blind cone이 메워짐을 전제로 함.
도입 경위·실측 근거는 HISTORY.md §1 참고.

**진입(entry)**: coverage 시작 지점은, 시작하자마자 그 스와스 전체 길이만큼
계속 멀어지며 캡처가 이어지므로(시간제한 없음) 대체로 이 원칙을 자연히
만족함. 유일한 예외는 미션 전체의 진짜 첫 coverage 지점(p0) - 그 이전에
로봇이 존재한 적이 없어 "접근" 쪽 시야 자체가 원천적으로 없음. 이 한 지점만
run_start_prepass가 인위적으로 접근 시야를 만들어줌 - 로봇 자체가 p0가
아니라 p0보다 진행방향으로 boundary_repass_distance_m만큼 앞선 '러닝스타트
지점'에서 스폰/배치되고(MissionExecutor._compute_mission_start_pose가 계산,
sim은 텔레포트 좌표로, real은 AMCL 초기 위치 힌트로 사용), run_start_prepass는
그 지점에서 캡처를 켠 채로 p0까지 주행해 들어가는 동작 하나만 수행함.

**이탈(exit)**: coverage 구간이 끝나는 모든 지점(중간 노드 exit 포함, 미션의
진짜 마지막 지점도 포함)이 구조적으로 "멀어짐" 쪽 시야가 보장되지 않음.
run_exit_repass가 모든 coverage exit에서 동일하게, 왔던 방향으로 짧게
되짚어 재통과시켜 반대 방향 시야를 명시적으로 보장함. 왕복 구간이 세그먼트
자체가 지나가는 길 안에 있으므로 별도의 벽 근접/안전마진 계산이 필요 없음.

blind cone이 없는 센서로 교체되면 이 보정 자체가 불필요해질 수 있음(단,
아래 클래스 docstring 및 CLAUDE.md의 "향후 센서 교체 시 주의" 참고 - 검증
없이 가정하지 말 것) - `mission_execution.enable_boundary_repass`를 false로
끄면 MissionExecutor 본체 로직은 건드리지 않고 이 파일의 동작만 완전히
비활성화됨(캡처는 정상 진행하되 왕복 없이 즉시 시작/종료로 폴백).
"""

import math

from geometry_msgs.msg import PoseStamped


class BoundaryRepassController:
    """
    run_start_prepass는 미션의 진짜 첫 sub-segment 시작 전(로봇이 이미
    러닝스타트 지점에서 시작한 상태로, 거기서 p0까지 캡처를 켠 채 주행해
    들어가는 동작) 딱 한 번만 호출됨. run_exit_repass는 모든 coverage
    exit 경계(중간 노드 exit + 미션의 진짜 마지막 지점)마다 매번 호출됨 -
    되짚어 후진 방향으로 재통과한 뒤 캡처를 종료하고, 로봇은 원래 exit
    지점보다 약간 안쪽에 남음. 뒤따르는 transit sub-segment는 로봇의 실제
    현재 위치부터 경로를 새로 짜므로, 복귀 동작 없이 정상 진행됨.

    executor(MissionExecutor)가 이미 가진 실행 primitive(회전/직선주행/캡처
    서비스 호출)를 그대로 재사용함 - 이 클래스 자체는 그 primitive들을 어떤
    순서로 왕복 조합하는지에 대한 안무(choreography)만 담당함. 상태 없는
    순수 함수 모음인 다른 utils/ 파일과 달리 클래스로 만든 이유, 그리고
    되짚기/프리패스가 리터럴 후진이 아니라 유턴+전진인 이유는 CLAUDE.md의
    "진입 vs 이탈의 비대칭" 항목 참고.
    """

    def __init__(self, executor):
        self._exec = executor

    # ------------------------------------------------------------------
    # 공용 기하 헬퍼
    # ------------------------------------------------------------------

    def _segment_heading_rad(self, seg_poses):
        """세그먼트의 진행 방향(양 끝점을 잇는 벡터)을 반환함. 두 끝점이 사실상
        같은 점이면(고립 1점 등) 그 점에 저장된 orientation으로 대체함."""
        p0 = seg_poses[0].pose.position
        p1 = seg_poses[-1].pose.position
        dx = p1.x - p0.x
        dy = p1.y - p0.y
        if math.hypot(dx, dy) > 1e-3:
            return math.atan2(dy, dx)
        q = seg_poses[0].pose.orientation
        return 2.0 * math.atan2(q.z, q.w)

    def _offset_pose(self, base_pose, heading_rad, distance_m):
        """base_pose 위치에서 heading_rad 방향으로 distance_m만큼(음수면 반대
        방향으로) 떨어진 지점을 가리키는 PoseStamped를 만듦."""
        new_pose = PoseStamped()
        new_pose.header.frame_id = 'map'
        new_pose.header.stamp = self._exec.navigator.get_clock().now().to_msg()
        new_pose.pose.position.x = base_pose.pose.position.x + distance_m * math.cos(heading_rad)
        new_pose.pose.position.y = base_pose.pose.position.y + distance_m * math.sin(heading_rad)
        new_pose.pose.position.z = 0.0
        new_pose.pose.orientation.z = math.sin(heading_rad / 2.0)
        new_pose.pose.orientation.w = math.cos(heading_rad / 2.0)
        return new_pose

    def compute_runway_pose(self, seg_poses):
        """미션 시작 sub-segment(seg_poses, 첫 coverage sub-segment)로부터
        러닝스타트 지점을 계산해서 반환함 - p0(seg_poses[0])보다 진행방향
        으로 _repass_distance_m(seg_poses)만큼 앞선 위치, orientation은 p0를
        향하는 방향(진행방향의 반대). MissionExecutor._compute_mission_start_pose
        가 로봇의 실제 스폰(sim)/AMCL 초기 위치(real) 계산에 사용함 - 로봇이
        여기서 시작해야 run_start_prepass가 이 지점에서 p0로 캡처를 켠 채
        들어가는 것만으로 프리패스가 끝남. 세그먼트가 너무 짧아 의미 있는
        왕복 거리를 못 만들면 None을 반환함."""
        if len(seg_poses) < 2:
            return None
        d = self._repass_distance_m(seg_poses)
        if d < 0.3:
            return None

        p0 = seg_poses[0]
        heading = self._segment_heading_rad(seg_poses)
        runway_pose = self._offset_pose(p0, heading, d)
        face_p0 = heading + math.pi
        runway_pose.pose.orientation.z = math.sin(face_p0 / 2.0)
        runway_pose.pose.orientation.w = math.cos(face_p0 / 2.0)
        return runway_pose

    def _repass_distance_m(self, seg_poses):
        """설정값과 세그먼트 실제 길이 중 작은 쪽으로 왕복 거리를 clamp함 -
        방/구석이 왕복 거리보다 짧아 반대쪽 끝을 넘어가 버리는 것을 방지."""
        cfg = self._exec.mission_exec_cfg
        configured = cfg.get('boundary_repass_distance_m', 1.5)
        p0 = seg_poses[0].pose.position
        p1 = seg_poses[-1].pose.position
        seg_len = math.hypot(p1.x - p0.x, p1.y - p0.y)
        return min(configured, seg_len * 0.9)

    # ------------------------------------------------------------------
    # 미션 시작 프리패스: 로봇은 이미 러닝스타트 지점(runway point)에서
    # 시작한다는 것을 전제함 - MissionExecutor._compute_mission_start_pose가 이
    # 지점을 계산해서 스폰(sim)/AMCL 초기 위치(real)로 이미 써버렸기 때문임.
    # 그래서 이 함수는 캡처를 켠 채로 p0까지 주행해 들어가는 동작 하나만
    # 수행함. 이후 execute_mission()의 정상 첫 세그먼트 실행이 이어받음
    # (캡처가 이미 켜져 있으므로 _execute_capture_subsegment는 재시작 없이
    # 그대로 진행).
    # ------------------------------------------------------------------

    def run_start_prepass(self, seg_poses):
        ex = self._exec
        cfg = ex.mission_exec_cfg
        if not cfg.get('enable_boundary_repass', True):
            return
        if len(seg_poses) < 2:
            print("  [BoundaryRepass] Start segment too short for a prepass. Skipping.")
            return

        p0 = seg_poses[0]
        d = self._repass_distance_m(seg_poses)
        if d < 0.3:
            print(f"  [BoundaryRepass] Start segment too short for a meaningful prepass "
                  f"(clamped distance={d:.2f}m). Skipping.")
            return

        print(f"  [BoundaryRepass] Mission start: driving in from the runway point ({d:.2f}m) "
              f"to the true coverage start for two-sided blind-cone fill.")

        if not ex._rotate_in_place_to(p0, label='boundary_repass:start_prepass'):
            print("  [BoundaryRepass] Warning: prepass rotation failed. Falling back to normal start.")
            return

        started = ex._call_capture_service(ex.start_capture_client, "start_waypoint_capture (boundary prepass)")
        if not started:
            print("  [BoundaryRepass] Warning: capture start failed during prepass. Falling back to normal start.")
            return
        ex._capture_active = True

        if not ex._navigate_to_pose_blocking(p0, label='boundary_repass:start_prepass'):
            print("  [BoundaryRepass] Warning: prepass drive-in failed. "
                  "Capture stays on; continuing mission from wherever the robot ended up.")

        print("  [BoundaryRepass] Prepass complete. Resuming normal mission start with capture already active.")

    # ------------------------------------------------------------------
    # coverage exit 리패스: 정상 도착 후 캡처를 바로 끄지 않고, 왔던 방향으로
    # 짧게 되짚어 후진 방향으로 재통과한 뒤에 끔. 호출 시점에 이미
    # self._capture_active가 True임을 전제함(execute_mission()이 매 coverage
    # exit 경계마다 그 조건 하에서 호출 - 중간 노드 exit과 미션의 진짜 마지막
    # 지점 모두 동일하게 처리됨). 되짚기 후 로봇은 seg_poses[-1](원래 exit
    # 지점)보다 약간 안쪽에 남지만, 뒤따르는 transit sub-segment가 로봇의
    # 실제 위치부터 경로를 새로 짜므로 이 함수가 따로 복귀시킬 필요는 없음.
    # ------------------------------------------------------------------

    def run_exit_repass(self, seg_poses):
        ex = self._exec
        cfg = ex.mission_exec_cfg
        if not cfg.get('enable_boundary_repass', True):
            ex._call_capture_service(ex.stop_capture_client, "stop_waypoint_capture")
            ex._capture_active = False
            return
        if len(seg_poses) < 2:
            print("  [BoundaryRepass] Exit segment too short for a repass. Stopping capture immediately.")
            ex._call_capture_service(ex.stop_capture_client, "stop_waypoint_capture")
            ex._capture_active = False
            return

        p_end = seg_poses[-1]
        heading = self._segment_heading_rad(seg_poses)
        d = self._repass_distance_m(seg_poses)
        if d < 0.3:
            print(f"  [BoundaryRepass] Exit segment too short for a meaningful repass "
                  f"(clamped distance={d:.2f}m). Stopping capture immediately.")
            ex._call_capture_service(ex.stop_capture_client, "stop_waypoint_capture")
            ex._capture_active = False
            return

        print(f"  [BoundaryRepass] Coverage exit: retracing last {d:.2f}m for two-sided blind-cone fill.")
        retrace_pose = self._offset_pose(p_end, heading, -d)

        if not ex._rotate_in_place_to(retrace_pose, label='boundary_repass:exit_repass'):
            print("  [BoundaryRepass] Warning: repass rotation failed. Stopping capture as-is.")
        elif not ex._navigate_to_pose_blocking(retrace_pose, label='boundary_repass:exit_repass'):
            print("  [BoundaryRepass] Warning: repass retrace drive failed. Stopping capture as-is.")
        else:
            print("  [BoundaryRepass] Repass complete.")

        ex._call_capture_service(ex.stop_capture_client, "stop_waypoint_capture")
        ex._capture_active = False
