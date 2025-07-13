#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MoveOnSlidingSphere – 逐次ループ版（面内制約付き / 完全実行可）
--------------------------------------------------------------
* EE‑Z 軸は常に塗装面（y = const.）内に投影（dir_y = 0）
* θ = 3° 刻みスネーク走査 → 法線角度は単調変化
* 各ウェイポイントを逐次 CartesianPath + FollowJointTrajectory で実行
* マーカーで球中心・EE 轨跡を可視化
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import tf_transformations

from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import JointState

from moveit_msgs.srv import GetPositionIK, GetCartesianPath
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory

from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from rclpy.duration import Duration

from visualization_msgs.msg import Marker, MarkerArray


class MoveOnSlidingSphere(Node):
    """逐次実行で面内制約を守って球面を滑らかに塗装する"""

    def __init__(self):
        super().__init__("move_on_sliding_sphere")

        # ───────────── pubs / subs / TF ─────────────
        self.marker_pub = self.create_publisher(Marker, "/visualization_marker", 10)
        self.marker_array_pub = self.create_publisher(MarkerArray, "/visualization_marker_array", 10)

        self.ee_traj = []  # EE 轨跡保存
        self.create_timer(0.2, self.update_ee_marker)

        self.current_joint_state = JointState()
        self.create_subscription(JointState, "/joint_states", self.joint_state_cb, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ───────────── parameters ─────────────
        self.center_base = [0.0, -0.40, 0.1]  # 球中心（基準）
        self.radius = 0.15  # 球半径 [m]
        self.y_planes = [-0.40, -0.39, -0.38, -0.25]  # 断面 y 値
        self.theta_step_deg = 3.0  # θ 刻み [deg]
        self.max_slide = 0.15  # スライド量 [m] (球半径と同じ値がよい)

        # ───────────── service / action clients ─────────────
        self.ik_cli = self.wait_srv(GetPositionIK, "/compute_ik")
        self.cart_cli = self.wait_srv(GetCartesianPath, "/compute_cartesian_path")
        self.traj_cli = ActionClient(self, FollowJointTrajectory,
                                     "/joint_trajectory_controller/follow_joint_trajectory")

        # ───────────── build way‑points ─────────────
        self.waypoints = self.build_waypoints()
        self.get_logger().info(f"✅ {len(self.waypoints)} feasible poses")
        for _, c in self.waypoints:
            self.publish_sphere_marker(c)

        # ───────────── start loop ─────────────
        self.curr_index = 0
        self.wait_for_joint_state()
        self.execute_next_segment()

    # ============================================================ utils ===
    def wait_srv(self, srv_type, name):
        cli = self.create_client(srv_type, name)
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for {name} …")
        return cli

    def joint_state_cb(self, msg):
        self.current_joint_state = msg

    def wait_for_joint_state(self):
        while not self.current_joint_state.name:
            rclpy.spin_once(self)

    # ============================================ pose generation ========
    def build_waypoints(self):
        poses = []
        step = int(self.theta_step_deg)
        for idx, y in enumerate(self.y_planes):
            dy = y - self.center_base[1]
            if abs(dy) > self.radius:
                continue
            circle_r = math.sqrt(self.radius ** 2 - dy ** 2)
            thetas = range(0, 181, step) if idx % 2 == 0 else range(180, -1, -step)
            for t_deg in thetas:
                theta = math.radians(t_deg)
                slide = -self.max_slide * (1 - math.cos(theta)) / 2
                cx = self.center_base[0] + slide
                x = cx + circle_r * math.cos(theta)
                z = self.center_base[2] + circle_r * math.sin(theta)
                pose = self.compute_pose_plane(x, y, z, cx)
                if self.ik_ok(pose):
                    poses.append((pose, [cx, self.center_base[1], self.center_base[2]]))
        return poses

    def compute_pose_plane(self, x, y, z, cx):
        dir_x, dir_z = cx - x, self.center_base[2] - z
        norm = math.hypot(dir_x, dir_z)
        dir_x, dir_z = dir_x / norm, dir_z / norm
        up = [0, 1, 0]
        x_axis = [up[1] * dir_z, -up[0] * dir_z, up[0] * dir_x]
        x_axis = [v / math.sqrt(sum(k * k for k in x_axis)) for v in x_axis]
        y_axis = [-dir_z * x_axis[1], dir_z * x_axis[0] - dir_x * x_axis[2], dir_x * x_axis[1]]
        rot = [[x_axis[0], y_axis[0], dir_x, 0],
               [x_axis[1], y_axis[1], 0.0, 0],
               [x_axis[2], y_axis[2], dir_z, 0],
               [0, 0, 0, 1]]
        q = tf_transformations.quaternion_from_matrix(rot)
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = "base_link"
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = q
        return ps

    def ik_ok(self, pose):
        req = GetPositionIK.Request()
        req.ik_request.group_name = "ar_manipulator"
        req.ik_request.pose_stamped = pose
        req.ik_request.ik_link_name = "ee_link"
        req.ik_request.timeout.sec = 1
        fut = self.ik_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
        return fut.result() and fut.result().error_code.val == 1

    # ========================================== sequential execution =====
    def execute_next_segment(self):
        if self.curr_index >= len(self.waypoints):
            self.get_logger().info("🎉 Finished all waypoints")
            rclpy.shutdown(); return

        pose, _ = self.waypoints[self.curr_index]
        self.get_logger().info(f"▶️ Segment {self.curr_index + 1}/{len(self.waypoints)}")

        cart_req = GetCartesianPath.Request()
        cart_req.header.frame_id = "base_link"
        cart_req.group_name = "ar_manipulator"
        cart_req.waypoints = [pose.pose]
        cart_req.max_step = 0.005
        cart_req.jump_threshold = 0.0
        cart_req.avoid_collisions = True
        cart_req.start_state.joint_state = self.current_joint_state

        cart_fut = self.cart_cli.call_async(cart_req)
        rclpy.spin_until_future_complete(self, cart_fut, timeout_sec=5.0)
        res = cart_fut.result()
        if not res or res.error_code.val != 1 or res.fraction < 0.9:
            self.get_logger().error("🚫 Cartesian path segment failed")
            rclpy.shutdown(); return

        traj = res.solution.joint_trajectory
        dt = 0.25
        for i, pt in enumerate(traj.points):
            pt.time_from_start = Duration(seconds=dt * i).to_msg()

        goal = FollowJointTrajectory.Goal(); goal.trajectory = traj
        self.traj_cli.wait_for_server()
        send_fut = self.traj_cli.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_fut)
        gh = send_fut.result()
        if not gh or not gh.accepted:
            self.get_logger().error("🚫 Trajectory goal rejected"); rclpy.shutdown(); return

        res_fut = gh.get_result_async()
        rclpy.spin_until_future_complete(self, res_fut)
        self.get_logger().info(f"🌟 segment result {res_fut.result().error_code}")

        self.curr_index += 1
        self.execute_next_segment()

    # ========================================= markers & traces ==========
    def publish_sphere_marker(self, center):
        m = Marker()
        m.header.frame_id = "base_link"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "sphere"; m.id = 1000 + self.curr_index
        m.type, m.action = Marker.SPHERE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = center
        m.scale.x = m.scale.y = m.scale.z = self.radius * 2
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.5, 1.0, 0.4
        self.marker_pub.publish(m)

    def update_ee_marker(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                "base_link", "ee_link", Time(), timeout=Duration(seconds=0.2))
            p = trans.transform.translation
            self.ee_traj.append(Point(x=p.x, y=p.y, z=p.z))
            line = Marker()
            line.header.frame_id = "base_link"
            line.header.stamp = self.get_clock().now().to_msg()
            line.ns = "ee_path"; line.id = 0
            line.type, line.action = Marker.LINE_STRIP, Marker.ADD
            line.scale.x = 0.005; line.color.g, line.color.a = 1.0, 1.0
            line.points = self.ee_traj
            self.marker_array_pub.publish(MarkerArray(markers=[line]))
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = MoveOnSlidingSphere()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
