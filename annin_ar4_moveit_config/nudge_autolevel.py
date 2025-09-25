#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nudge_autolevel.py
- Arduino(MMA8452)の roll/pitch をUSBシリアルから読み取り
- 指定軸の平均角(目標0deg)から相対補正量(deg)を算出
- ROS2 FollowJointTrajectory で joint_trajectory_controller へ送信
"""
import argparse
import math
import re
import sys
import time
from typing import List, Optional

import serial  # apt: python3-serial / pip: pyserial
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# ===== デフォルト設定 =====
DEFAULT_JOINTS = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6']
DEFAULT_ONLY   = ['joint_3']  # 基本はJ3だけ補正
ACTION_NAME    = '/joint_trajectory_controller/follow_joint_trajectory'

# ------- ROS2 アクションクライアント -------
class JTCClient(Node):
    def __init__(self, action_name=ACTION_NAME):
        super().__init__('autolevel_nudge')
        self.client = ActionClient(self, FollowJointTrajectory, action_name)

    def wait(self, timeout=10.0):
        if not self.client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError('FollowJointTrajectory action server not available')

    def send(self, joint_names: List[str], target_rad: List[float], duration: float):
        traj = JointTrajectory()
        traj.joint_names = joint_names

        pt = JointTrajectoryPoint()
        pt.positions = target_rad
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration - int(duration))*1e9)
        traj.points.append(pt)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        goal.goal_time_tolerance.sec = 1

        self.get_logger().info(
            f'Sending target(rad) to {joint_names}: {["%.6f"%x for x in target_rad]} in {duration:.3f}s'
        )
        send_fut = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)
        gh = send_fut.result()
        if not gh or not gh.accepted:
            raise RuntimeError('Goal rejected by controller')

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        res = res_fut.result().result
        self.get_logger().info(f'Result error_code: {res.error_code}')

# ------- センサ読取（roll,pitch） -------
ROLL_PITCH_CSV = re.compile(r'^\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*$')
def parse_roll_pitch(line: str):
    """
    受理フォーマット:
      1) "roll,pitch" （推奨CSV）
      2) "ax:.. ay:.. az:.. | roll:.. pitch:.."
    戻り: (roll_deg, pitch_deg) または None
    """
    m = ROLL_PITCH_CSV.match(line)
    if m:
        roll = float(m.group(1)); pitch = float(m.group(3))
        return roll, pitch

    if '| ' in line or '|' in line:
        try:
            right = line.split('|', 1)[1]
            right = right.replace('roll:', '').replace('pitch:', ' ').strip()
            parts = right.split()
            if len(parts) >= 2:
                roll = float(parts[0]); pitch = float(parts[1])
                return roll, pitch
        except Exception:
            return None
    return None

def read_sensor_average(port: str, baud: int, axis: str, samples: int, warmup_ms: int) -> float:
    """センサから複数サンプルを読み平均（deg）を返す。axis='pitch' or 'roll'"""
    with serial.Serial(port, baud, timeout=1) as ser:
        # ウォームアップ（バナー行などを流す）
        t0 = time.time()
        while (time.time() - t0) * 1000 < warmup_ms:
            ser.readline()

        vals = []
        t0 = time.time()
        while len(vals) < samples and (time.time() - t0) < 5.0:
            s = ser.readline().decode(errors='ignore').strip()
            if not s:
                continue
            rp = parse_roll_pitch(s)
            if rp is None:
                continue
            roll, pitch = rp
            vals.append(pitch if axis == 'pitch' else roll)
        if not vals:
            raise RuntimeError('No sensor samples parsed. Check format or port.')
        return sum(vals) / len(vals)

# ------- メイン -------
def main():
    p = argparse.ArgumentParser(description='Auto-level J3 using MMA8452 over USB and send ROS2 nudge.')
    p.add_argument('--port', default='/dev/ttyACM0', help='Arduino serial port (e.g., /dev/ttyACM0)')
    p.add_argument('--baud', type=int, default=115200)
    p.add_argument('--axis', choices=['pitch','roll'], default='pitch',
                   help='Which angle to drive to 0 deg (sensor mounting dependent)')
    p.add_argument('--samples', type=int, default=50, help='Number of samples to average')
    p.add_argument('--warmup-ms', type=int, default=600, help='Discard lines for this warmup time')
    p.add_argument('--only', nargs='+', default=DEFAULT_ONLY,
                   help='Target joints (default: joint_3)')
    p.add_argument('--all-joints', action='store_true',
                   help='Apply delta to DEFAULT_JOINTS order (mostly for testing)')
    p.add_argument('--duration', type=float, default=3.0, help='Seconds for the move')
    p.add_argument('--gain', type=float, default=1.0, help='delta = -gain * avg_angle')
    p.add_argument('--invert', action='store_true', help='Flip sign (if sensor orientation opposite)')
    p.add_argument('--deadband', type=float, default=0.05, help='|avg| below this [deg] -> send 0')
    p.add_argument('--limit', type=float, default=10.0, help='Clamp |delta| <= limit [deg]')
    p.add_argument('--dry-run', action='store_true', help='Compute & print but do not send')
    args = p.parse_args()

    # 1) センサ平均角[deg]
    avg = read_sensor_average(args.port, args.baud, args.axis, args.samples, args.warmup_ms)
    # 2) 目標0degへの相対補正（符号は取り付け向きで調整）
    delta = -args.gain * avg
    if args.invert:
        delta = -delta
    if abs(avg) < args.deadband:
        delta = 0.0
    # クランプ
    if abs(delta) > args.limit:
        delta = math.copysign(args.limit, delta)

    print(f"[sensor] {args.axis} avg = {avg:.3f} deg")
    print(f"[nudge ] delta = {delta:+.3f} deg  (deadband={args.deadband}, limit={args.limit}, gain={args.gain})")

    # 3) 送信準備（相対移動：deltaをそのまま“目標姿勢”として使用）
    #    → シンプルに「現在から delta だけ動かす」指示にしたいので、
    #       ここでは“相対専用”の簡易目標：JTCへは現在角に delta を足した姿勢を送る必要がある。
    #       ただし最小化のため、ここでは「対象関節以外は現在角を維持」の1点指令を送るため、
    #       JTC側のホールド/位置保持が効く構成を前提とします。
    #    実機の構成により“現在角の取得（/joint_states）→加算”が必要な場合は拡張してください。
    joints = DEFAULT_JOINTS if args.all_joints else args.only
    # 対象関節だけにdelta、他は0を送る（JTCのinterpretationに依存します）
    target_deg = [0.0]*len(joints)
    # joint_3 が含まれていないなら最初の要素に適用（保険）
    apply_idx = joints.index('joint_3') if 'joint_3' in joints else 0
    target_deg[apply_idx] = delta
    # deg->rad
    target_rad = [math.radians(x) for x in target_deg]

    print(f"[send  ] joints={joints}")
    print(f"[send  ] target(rad)={['%.6f'%v for v in target_rad]}  duration={args.duration:.2f}s")
    if args.dry_run:
        return

    # 4) 送信
    rclpy.init()
    node = JTCClient()
    try:
        node.wait()
        node.send(joints, target_rad, args.duration)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
