#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nudge_autolevel_abs.py
- Arduino(MMA8452)の roll/pitch をUSBシリアルから読み取り
- 平均角(avg)から delta = -gain * avg を算出（deadband/limit付き）
- /joint_states を読み「現在角＋delta」を“絶対角”として
  /joint_trajectory_controller/follow_joint_trajectory に送信
"""

import argparse, math, re, time, sys
from typing import List, Optional, Tuple

import serial  # apt: python3-serial / pip: pyserial

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

DEFAULT_JOINTS = ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6']
ACTION_NAME = '/joint_trajectory_controller/follow_joint_trajectory'

# -------------------- センサ読み取り --------------------
ROLL_PITCH_CSV = re.compile(r'^\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*$')

def parse_roll_pitch(line: str) -> Optional[Tuple[float,float]]:
    """
    受理フォーマット:
      1) "roll,pitch" （CSV推奨）
      2) "ax:.. ay:.. az:.. | roll:.. pitch:.."
    戻り: (roll_deg, pitch_deg) or None
    """
    m = ROLL_PITCH_CSV.match(line)
    if m:
        return float(m.group(1)), float(m.group(3))
    if '|' in line:
        try:
            right = line.split('|',1)[1]
            right = right.replace('roll:','').replace('pitch:',' ')
            parts = right.split()
            if len(parts)>=2:
                return float(parts[0]), float(parts[1])
        except Exception:
            return None
    return None

def read_sensor_average(port: str, baud: int, axis: str, samples: int, warmup_ms: int) -> float:
    """axis='pitch' or 'roll' を平均[deg]で返す"""
    with serial.Serial(port, baud, timeout=1) as ser:
        t0=time.time()
        while (time.time()-t0)*1000 < warmup_ms:
            ser.readline()

        vals=[]
        t0=time.time()
        while len(vals)<samples and (time.time()-t0)<5.0:
            s = ser.readline().decode(errors='ignore').strip()
            if not s: continue
            rp = parse_roll_pitch(s)
            if rp is None: continue
            roll, pitch = rp
            vals.append(pitch if axis=='pitch' else roll)
        if not vals:
            raise RuntimeError('No sensor samples parsed. Check format/port.')
        return sum(vals)/len(vals)

# -------------------- /joint_states 1回読み --------------------
class OneShotJointState(Node):
    def __init__(self, target_names: List[str], timeout: float):
        super().__init__('oneshot_joint_state_reader')
        self.target_names = target_names
        self.timeout = timeout
        self.sub = self.create_subscription(JointState, '/joint_states', self._cb, 10)
        self._got=False
        self.positions=None

    def _cb(self, msg: JointState):
        m = dict(zip(msg.name, msg.position))
        if all(n in m for n in self.target_names):
            self.positions = [m[n] for n in self.target_names]
            self._got=True

    def read_once(self) -> Optional[List[float]]:
        start = self.get_clock().now()
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._got:
                return self.positions
            if (self.get_clock().now() - start).nanoseconds * 1e-9 > self.timeout:
                return None
        return None

# -------------------- JTC クライアント --------------------
class JTCClient(Node):
    def __init__(self, action_name=ACTION_NAME):
        super().__init__('autolevel_nudge_abs')
        self.client = ActionClient(self, FollowJointTrajectory, action_name)

    def wait(self, timeout=10.0):
        if not self.client.wait_for_server(timeout_sec=timeout):
            raise RuntimeError('FollowJointTrajectory action server not available')

    def send_abs(self, joint_names: List[str], target_rad: List[float], duration: float):
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
            f'Sending ABS to {joint_names}: {["%.6f"%x for x in target_rad]} in {duration:.3f}s'
        )
        gf = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, gf)
        gh = gf.result()
        if not gh or not gh.accepted:
            raise RuntimeError('Goal rejected by controller')

        rf = gh.get_result_async()
        rclpy.spin_until_future_complete(self, rf)
        res = rf.result().result
        self.get_logger().info(f'Result error_code: {res.error_code}')

# -------------------- メイン --------------------
def main():
    ap = argparse.ArgumentParser(description='Auto-level J3 (ABS) using MMA8452 and JTC.')
    ap.add_argument('--port', default='/dev/ttyACM0', help='Arduino serial port')
    ap.add_argument('--baud', type=int, default=115200)
    ap.add_argument('--axis', choices=['pitch','roll'], default='pitch')
    ap.add_argument('--samples', type=int, default=50)
    ap.add_argument('--warmup-ms', type=int, default=600)
    ap.add_argument('--joints', nargs='+', default=DEFAULT_JOINTS,
                    help='All controller joints in order (ABS指令用に必須)')
    ap.add_argument('--target-only', nargs='+', default=['joint_3'],
                    help='補正したい関節（既定: joint_3）')
    ap.add_argument('--duration', type=float, default=3.0)
    ap.add_argument('--gain', type=float, default=1.0)
    ap.add_argument('--invert', action='store_true')
    ap.add_argument('--deadband', type=float, default=0.05)
    ap.add_argument('--limit', type=float, default=10.0)
    ap.add_argument('--jstates-timeout', type=float, default=2.0)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    # 1) センサ平均角[deg] → delta[deg]
    avg = read_sensor_average(args.port, args.baud, args.axis, args.samples, args.warmup_ms)
    delta = -args.gain * avg
    if args.invert:
        delta = -delta
    if abs(avg) < args.deadband:
        delta = 0.0
    if abs(delta) > args.limit:
        delta = math.copysign(args.limit, delta)

    print(f"[sensor] {args.axis} avg = {avg:.3f} deg")
    print(f"[delta ] {delta:+.3f} deg (deadband={args.deadband}, limit={args.limit}, gain={args.gain})")

    # 2) /joint_states から現在“絶対角[rad]”を取得し、ターゲット生成
    rclpy.init()
    reader = OneShotJointState(args.joints, args.jstates-timeout if hasattr(args,'jstates-timeout') else args.jstates_timeout)
    cur = reader.read_once()
    reader.destroy_node()
    if cur is None:
        rclpy.shutdown()
        print("[ERROR] Could not read /joint_states. Make sure joint_state_broadcaster is active.", file=sys.stderr)
        sys.exit(2)

    # 3) 対象関節に delta[deg] を加算（→ ABS[rad]）
    target = list(cur)  # rad
    delta_rad = math.radians(delta)
    for name in args.target_only:
        if name not in args.joints:
            rclpy.shutdown()
            print(f"[ERROR] target-only joint '{name}' not in controller joints {args.joints}", file=sys.stderr)
            sys.exit(2)
        idx = args.joints.index(name)
        target[idx] = cur[idx] + delta_rad

    print(f"[abs  ] joints={args.joints}")
    print(f"[abs  ] cur(rad)   = {['%.6f'%v for v in cur]}")
    print(f"[abs  ] target(rad)= {['%.6f'%v for v in target]}  duration={args.duration:.2f}s")
    if args.dry_run:
        rclpy.shutdown()
        return

    # 4) 送信（ABS）
    node = JTCClient()
    try:
        node.wait()
        node.send_abs(args.joints, target, args.duration)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
