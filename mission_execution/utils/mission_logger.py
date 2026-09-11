# mission_execution/utils/mission_logger.py

"""
주행 중 "로봇이 그 시점에 정확히 어디서/무엇을 하고 있었는지"를 일정 간격으로
(기본 3초, params.yaml의 drive_debug_interval_sec) 남기는 실측 디버깅용
주기 로거. 매 tick마다 파일을 flush하므로, 미션 도중 Ctrl+C나 프로세스
kill로 중단돼도 그 직전까지의 행(row)은 이미 디스크에 안전하게 남아있음 -
깔끔한 종료(stop())를 못 거쳐도 데이터 유실은 최대 한 tick分(기본 3초)뿐임.

stall_logger.py(StallWatcher)는 각 blocking 호출이 끝난 "이후"에야 그 구간의
정체 요약(시작 시각/길이/recoveries 여부)을 CSV 한 줄로 남김 - 미션 전체가
끝나야 확인 가능하고, "그 정체가 정확히 어느 노드/coverage/transit/prepass/
repass 구간에서 일어났는지"는 label 문자열 하나에 의존함. 방 한복판(벽 근접이
아닌 곳)에서도 멈추는 사례가 보고되면서, 사후 요약만으로는 원인 재구성이
어려워 이 모듈을 추가함 - 미션이 도는 동안 실시간으로 파일에 계속 쌓이므로
중간에 멈춰도(프로세스 kill 등) 그때까지의 기록은 남음.

스레드 안전성:
- 이 클래스는 MissionExecutor.tf_buffer를 직접 lookup만 함(spin_once를 절대
  부르지 않음). 메인 스레드가 이미 모든 blocking 대기 루프에서 0.01~0.05초
  간격으로 spin_executor.spin_once()를 돌리고 있어 버퍼는 그걸로 갱신되고,
  SingleThreadedExecutor를 두 스레드가 동시에 건드리면 위험하므로 백그라운드
  스레드에서는 읽기 전용 lookup_transform만 호출함(tf2 Buffer는 이 패턴을
  위해 내부적으로 스레드 세이프하게 설계됨).
- MissionExecutor._nav_status/_current_context는 메인 스레드가 항상 "새
  dict/문자열 통째로 재할당"만 함(제자리 수정 없음) - CPython GIL 하에서
  객체 참조 재할당은 원자적이므로, 백그라운드 스레드가 읽을 때 별도 락 없이도
  어중간하게 섞인 상태를 볼 일이 없음.
"""

import os
import csv
import math
import time
import threading

import rclpy


class MissionLogger:
    def __init__(self, executor, log_path, interval_sec=3.0, stall_threshold_sec=9.0):
        self._executor = executor
        self.log_path = log_path
        self.interval_sec = interval_sec
        self.stall_threshold_sec = stall_threshold_sec

        self._stop_event = threading.Event()
        self._thread = None
        self._mission_start = time.time()

        # 정체 판단용 상태 - (nav_action, progress_kind) 조합이 바뀌거나
        # progress_value가 유의미하게 움직이면 리셋됨.
        self._last_progress_key = None
        self._last_progress_value = None
        self._last_progress_time = self._mission_start
        self._stall_reported = False  # 이번 정체 구간에서 이미 사유를 콘솔에 찍었는지

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self._file = open(log_path, 'w', newline='')
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            'wall_time', 'mission_elapsed_sec', 'context',
            'x', 'y', 'yaw_deg', 'nav_action', 'progress_kind', 'progress_value',
            'number_of_recoveries', 'stalled_sec', 'note',
        ])
        self._file.flush()
        print(f"[*] Mission debug log: {log_path} (every {interval_sec:.0f}s, "
              f"stall detail threshold {stall_threshold_sec:.0f}s)")

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        try:
            self._file.close()
        except Exception:
            pass

    # ------------------------------------------------------------------

    def _get_pose(self):
        """읽기 전용 TF 조회 - spin_once를 부르지 않음(백그라운드 스레드 전용)."""
        try:
            trans = self._executor.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time())
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            yaw = self._executor._quaternion_to_yaw(trans.transform.rotation)
            return x, y, yaw
        except Exception:
            return None, None, None

    def _run(self):
        while not self._stop_event.wait(timeout=self.interval_sec):
            try:
                self._tick()
            except Exception as e:
                print(f"[MissionLogger] tick failed: {e}")

    def _tick(self):
        now = time.time()
        elapsed = now - self._mission_start
        context = getattr(self._executor, '_current_context', None) or 'unknown'
        x, y, yaw = self._get_pose()
        nav_status = getattr(self._executor, '_nav_status', None) or {}
        nav_action = nav_status.get('action')
        progress_kind = nav_status.get('progress_kind')
        progress_value = nav_status.get('progress_value')
        recoveries = nav_status.get('number_of_recoveries')

        progress_key = (nav_action, progress_kind)
        if progress_key != self._last_progress_key:
            # 새 액션이 시작됐거나 종류가 바뀜 - 정체 시계를 리셋함.
            self._last_progress_key = progress_key
            self._last_progress_value = progress_value
            self._last_progress_time = now
            self._stall_reported = False
        elif (progress_value is not None and self._last_progress_value is not None
                and abs(progress_value - self._last_progress_value) > 0.02):
            # 같은 액션이 유의미하게 진행 중 - 정체 시계를 리셋함.
            self._last_progress_value = progress_value
            self._last_progress_time = now
            self._stall_reported = False
        # progress_value가 계속 None인 경우(액션 없음/idle 포함)는 그 상태
        # 유지 자체가 self._last_progress_time을 안 갱신시켜 자연히 정체로 잡힘.

        stalled_sec = now - self._last_progress_time

        note = ''
        if stalled_sec >= self.stall_threshold_sec:
            note = self._build_stall_reason(
                context, nav_action, progress_kind, progress_value, recoveries, stalled_sec)
            if not self._stall_reported:
                print(f"[!] [MissionLogger] STALL {stalled_sec:.1f}s - {note}")
                self._stall_reported = True

        self._writer.writerow([
            time.strftime('%H:%M:%S', time.localtime(now)),
            f"{elapsed:.1f}",
            context,
            'None' if x is None else f"{x:.3f}",
            'None' if y is None else f"{y:.3f}",
            'None' if yaw is None else f"{math.degrees(yaw):.1f}",
            nav_action or 'idle',
            progress_kind or '',
            'None' if progress_value is None else f"{progress_value:.3f}",
            'None' if recoveries is None else recoveries,
            f"{stalled_sec:.1f}",
            note,
        ])
        self._file.flush()

    def _build_stall_reason(self, context, nav_action, progress_kind, progress_value,
                             recoveries, stalled_sec):
        """
        관측 가능한 신호(어떤 nav2 액션이 떠 있는지, 그 진행 지표가 실제로
        멈춰있는지, nav2 자체 recovery가 발동했는지)를 조합해 "경로 상에
        장애물이 있음" 수준을 넘어서는 구체적 1차 진단을 남김. 최종 확정은
        아니고, 이 로그를 보고 사람이 다음 확인 방향을 빨리 좁히기 위한 것.
        """
        if nav_action is None:
            return (f"Nav2 액션이 전혀 실행 중이지 않은 상태로 {stalled_sec:.1f}초 경과 "
                    f"(context={context}) - nav2 바깥(TF 조회 대기, capture 서비스 응답 "
                    f"대기, AMCL jump 처리 등 Python 쪽 로직)에서 멈춰있을 가능성이 큼. "
                    f"nav2/코스트맵 문제가 아닐 수 있음.")

        progress_str = 'N/A' if progress_value is None else f"{progress_value:.2f}"
        base = (f"{nav_action} 실행 중 {stalled_sec:.1f}초간 진행 없음 "
                f"(context={context}, {progress_kind}={progress_str}")
        base += '' if recoveries is None else f", number_of_recoveries={recoveries}"
        base += ")"

        if recoveries is not None and recoveries > 0:
            reason = (" -> Nav2 자체 recovery behavior(제자리 회전/후진 재시도 등, 기본 "
                      "BT의 RoundRobin recovery subtree)가 이미 이 구간에서 "
                      f"{recoveries}회 발동했는데도 진행이 없음. 단순 대기가 아니라 로컬 "
                      "planner/costmap이 반복적으로 유효한 경로를 못 찾고 있는 상태로 추정됨.")
        elif nav_action in ('Spin', 'BackUp'):
            reason = (" -> 제자리 회전/후진 액션 자체가 진행 지표를 전혀 못 만들고 "
                      "멈춰있음 - 시작 직후 사전 충돌 체크(simulate_ahead_time, "
                      "behavior_server)에 걸려 즉시 실패했거나, 명령은 전송됐지만 로봇이 "
                      "물리적으로 반응하지 않는 상태(하드웨어 쪽)일 수 있음.")
        else:
            reason = (" -> nav2 자체 recovery는 아직 발동 안 했는데도 진행이 없음 - "
                      "controller가 코스트맵상 근접 장애물 때문에 계속 0에 가까운 속도만 "
                      "계산 중이거나, 명령은 나가는데 로봇이 실제로 반응하지 않는 상태일 "
                      "가능성. /cmd_vel 값이 실제로 0인지 직접 확인 권장.")
        return base + reason
