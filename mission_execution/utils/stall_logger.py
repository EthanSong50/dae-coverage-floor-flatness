# mission_execution/utils/stall_logger.py

"""
주행 중 "왜, 얼마나 지연되는지" 디버깅용 정체(stall) 감지 유틸리티.

nav2 액션(NavigateToPose/NavigateThroughPoses/Spin)의 feedback을 blocking 대기
루프 안에서 계속 폴링하면서, 진행 지표(distance_remaining 또는
angular_distance_traveled)가 stall_threshold_sec 이상 전혀 변하지 않는 구간을
"정체"로 기록함. NavigateToPose/NavigateThroughPoses feedback의
number_of_recoveries가 그 구간 동안 증가했는지도 함께 기록해서, 정체가 nav2
자체의 recovery behavior(제자리 회전 재시도/costmap clear 등) 때문인지 여부를
가장 유력한 "왜"의 단서로 남김.

미션 전체를 도는 동안 여러 blocking 호출(_rotate_in_place_to/
_navigate_to_pose_blocking/_navigate_through_poses_blocking, boundary_repass의
호출 포함)에서 매번 이 클래스를 하나씩 새로 만들어 쓰고, finalize()로 수거한
이벤트를 MissionExecutor.execute_mission()이 self._stall_events에 누적함.
도입 배경은 HISTORY.md §4 참고.
"""

import time
import csv


class StallWatcher:
    def __init__(self, label, stall_threshold_sec=5.0, progress_eps=0.02):
        self.label = label
        self.stall_threshold_sec = stall_threshold_sec
        self.progress_eps = progress_eps
        self.start_time = time.time()

        self._last_progress_value = None
        self._last_progress_time = self.start_time
        self._last_recoveries = None
        self._stall_active = False
        self._stall_start_time = None
        self._stall_start_value = None
        self._recoveries_during_stall = 0
        self.events = []

    def update(self, progress_value, recoveries=None):
        now = time.time()

        if recoveries is not None:
            if self._stall_active and self._last_recoveries is not None and recoveries > self._last_recoveries:
                self._recoveries_during_stall += (recoveries - self._last_recoveries)
            self._last_recoveries = recoveries

        if self._last_progress_value is None:
            self._last_progress_value = progress_value
            self._last_progress_time = now
            return

        made_progress = (
            progress_value is not None
            and abs(progress_value - self._last_progress_value) > self.progress_eps
        )

        if made_progress:
            if self._stall_active:
                self._close_stall(now, resolved=True)
            self._last_progress_value = progress_value
            self._last_progress_time = now
            return

        stalled_for = now - self._last_progress_time
        if stalled_for >= self.stall_threshold_sec and not self._stall_active:
            self._stall_active = True
            self._stall_start_time = self._last_progress_time
            self._stall_start_value = self._last_progress_value

    def _close_stall(self, end_time, resolved):
        self.events.append({
            'label': self.label,
            'stall_start_abs': self._stall_start_time,
            'duration_sec': end_time - self._stall_start_time,
            'stalled_value': self._stall_start_value,
            'recoveries_during_stall': self._recoveries_during_stall,
            'resolved': resolved,
        })
        self._stall_active = False
        self._recoveries_during_stall = 0

    def finalize(self):
        """대기 루프가 끝나는 시점(성공/실패 무관)에 반드시 호출함 - 정체가 해소되지
        않은 채로 액션 자체가 끝난 경우(resolved=False)까지 포함해 이벤트를 반환함."""
        if self._stall_active:
            self._close_stall(time.time(), resolved=False)
        return self.events


def write_stall_report(events, output_path, mission_start_abs, threshold_sec):
    """수집된 정체 이벤트를 CSV 한 파일에 그것만 따로 저장함 - 다른 로그와
    섞이지 않아 grep/정렬로 바로 디버깅할 수 있게 하기 위함."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([f"# stall threshold = {threshold_sec:.1f}s, "
                          f"{len(events)} stall(s) detected"])
        writer.writerow([
            'context_label', 'mission_elapsed_sec_at_stall_start', 'duration_sec',
            'stalled_progress_value', 'recoveries_during_stall', 'resolved',
        ])
        for ev in sorted(events, key=lambda e: e['stall_start_abs']):
            writer.writerow([
                ev['label'],
                f"{ev['stall_start_abs'] - mission_start_abs:.1f}",
                f"{ev['duration_sec']:.1f}",
                'None' if ev['stalled_value'] is None else f"{ev['stalled_value']:.3f}",
                ev['recoveries_during_stall'],
                'yes' if ev['resolved'] else 'no (still stalled when action ended)',
            ])
