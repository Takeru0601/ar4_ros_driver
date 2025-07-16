#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
move_along_sphere_perpendicular_cartesian_with_moving_sphere_withsg.py

・球マーカーは最初のMoveIt実行（目標点1点目→2点目）までは「2点目目標点の法線上・z固定」にて完全固定
・3点目以降はスプレーガン先端(xのみ)に追従（y,zは初期値で固定）
・スプレーガン先端＝ee_linkから法線方向に0.188m
・球中心＝スプレーガン先端＋法線方向に球半径
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import tf2_ros
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from moveit_msgs.srv import GetPositionIK, GetCartesianPath
from moveit_msgs.action import MoveGroup, ExecuteTrajectory
from moveit_msgs.msg import (
    MotionPlanRequest,
    Constraints,
    JointConstraint,
    OrientationConstraint,
)
from builtin_interfaces.msg import Duration as DurationMsg
import csv
import time

def quaternion_to_rotmat(q):
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    return np.array([
        [1-2*(yy+zz),   2*(xy-wz),   2*(xz+wy)],
        [2*(xy+wz),   1-2*(xx+zz),   2*(yz-wx)],
        [2*(xz-wy),     2*(yz+wx), 1-2*(xx+yy)]
    ])

class MoveOnSphereFollowingX0(Node):
    def __init__(self):
        super().__init__("move_on_sphere_following_x0")

        self.joint_names   = [f"joint_{i}" for i in range(1,7)]
        self.cur_positions = [0.0]*6
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)

        self.marker_pub = self.create_publisher(Marker, "/visualization_marker", 10)
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.exec_pts = []

        self.sphere_radius   = 0.1    # m
        self.sphere_y        = -0.5
        self.sphere_z        =  0.2
        self.y_planes        = np.linspace(self.sphere_y, self.sphere_y+0.1, 5).tolist()
        self.spraygun_length = 0.188  # m (188mm)

        self.arc_points = self._generate_intersection_points_following_x0()
        # 2点目目標点法線上スプレーガン先端+球半径を初期球中心に
        if len(self.arc_points) > 1:
            pose_2nd, _, spraygun_tip_2nd = self.arc_points[1]
            quat = [
                pose_2nd.pose.orientation.x,
                pose_2nd.pose.orientation.y,
                pose_2nd.pose.orientation.z,
                pose_2nd.pose.orientation.w,
            ]
            rot_mat = quaternion_to_rotmat(quat)
            z_axis = rot_mat[:, 2]
            self.initial_sphere_center = spraygun_tip_2nd + z_axis * self.sphere_radius
        else:
            _, init_center, _ = self.arc_points[0]
            self.initial_sphere_center = init_center.copy()

        self.move_client = ActionClient(self, MoveGroup, "move_action")
        self.ik_client   = self.create_client(GetPositionIK, "compute_ik")
        self.idx = 0

        self.prev_time, self.prev_tip_pos = None, None
        self.speed_sum, self.speed_count = 0.0, 0
        self.csv_file = open("spraygun_tip_speed_log.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["time","x","y","z","speed(m/s)"])

        # マーカー初回描画＋タイマーで定期再描画
        self._pub_dynamic_sphere_marker()
        self._pub_exec_path_marker()
        self.create_timer(0.0005, self._pub_dynamic_sphere_marker)
        self.create_timer(0.0005, self._pub_exec_path_marker)
        self.create_timer(0.005, self._record_ee_position)  # 軌跡用

        self.get_logger().info(f"arc_points総数: {len(self.arc_points)}")
        self.get_logger().info("Waiting for MoveGroup action server …")
        self.move_client.wait_for_server()
        self.get_logger().info("Waiting for compute_ik service …")
        self.ik_client.wait_for_service()
        self.speed_scaling = 0.5

        self._send_first_point()

    def _js_cb(self, msg: JointState):
        d = dict(zip(msg.name, msg.position))
        try:
            self.cur_positions = [d[n] for n in self.joint_names]
        except KeyError:
            pass

    def _generate_intersection_points_following_x0(self):
        pts = []
        steps = 36
        for idx,y in enumerate(self.y_planes):
            dy = y - self.sphere_y
            if abs(dy) > self.sphere_radius: continue
            r = math.sqrt(self.sphere_radius**2 - dy**2)
            order = range(steps+1) if idx%2==0 else range(steps,-1,-1)
            for i in order:
                theta = math.radians(180.0 * i / steps)
                x_c = r * math.cos(theta)
                z_c = self.sphere_z + r * math.sin(theta)
                center = np.array([x_c, self.sphere_y, self.sphere_z])
                p = np.array([0.0, y, z_c])
                dirv = center - p
                dirv /= np.linalg.norm(dirv)
                up = np.array([0,1,0])
                x_axis = np.cross(up, dirv)
                x_axis /= np.linalg.norm(x_axis)
                y_axis = np.cross(dirv, x_axis)
                R = np.column_stack((x_axis, y_axis, dirv))
                q = self.rotation_matrix_to_quaternion(R)

                spray_tip = p.copy()
                ee_target = spray_tip - dirv * self.spraygun_length

                pose = PoseStamped()
                pose.header.frame_id = "base_link"
                pose.pose.position.x = float(ee_target[0])
                pose.pose.position.y = float(ee_target[1])
                pose.pose.position.z = float(ee_target[2])
                pose.pose.orientation.x = float(q[0])
                pose.pose.orientation.y = float(q[1])
                pose.pose.orientation.z = float(q[2])
                pose.pose.orientation.w = float(q[3])

                pts.append((pose, center, spray_tip))
        return pts

    def rotation_matrix_to_quaternion(self, m):
        tr = float(m[0,0]+m[1,1]+m[2,2])
        if tr>0:
            s = math.sqrt(tr+1.0)*2.0
            w=0.25*s; x=(m[2,1]-m[1,2])/s
            y=(m[0,2]-m[2,0])/s; z=(m[1,0]-m[0,1])/s
        elif m[0,0]>m[1,1] and m[0,0]>m[2,2]:
            s = math.sqrt(1.0+m[0,0]-m[1,1]-m[2,2])*2.0
            w=(m[2,1]-m[1,2])/s; x=0.25*s
            y=(m[0,1]+m[1,0])/s; z=(m[0,2]+m[2,0])/s
        elif m[1,1]>m[2,2]:
            s = math.sqrt(1.0+m[1,1]-m[0,0]-m[2,2])*2.0
            w=(m[0,2]-m[2,0])/s; x=(m[0,1]+m[1,0])/s
            y=0.25*s; z=(m[1,2]+m[2,1])/s
        else:
            s = math.sqrt(1.0+m[2,2]-m[0,0]-m[1,1])*2.0
            w=(m[1,0]-m[0,1])/s; x=(m[0,2]+m[2,0])/s
            y=(m[1,2]+m[2,1])/s; z=0.25*s
        return np.array([x,y,z,w], dtype=float)

    def _pub_dynamic_sphere_marker(self):
        fixed_y = self.initial_sphere_center[1]
        fixed_z = self.initial_sphere_center[2]

        if self.idx < 2:
            # 2点目の球中心（法線上・z固定）で固定
            sc = self.initial_sphere_center.copy()
        else:
            try:
                tf = self.tf_buffer.lookup_transform("base_link","ee_link",rclpy.time.Time())
                ee_pos = np.array([
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z
                ])
                q = tf.transform.rotation
                quat = [q.x, q.y, q.z, q.w]
                rot_mat = quaternion_to_rotmat(quat)
                z_axis = rot_mat[:, 2]
                # スプレーガン先端 = ee_link + z_axis * 0.188
                spraygun_tip = ee_pos + z_axis * self.spraygun_length
                # 球中心 = スプレーガン先端 + z_axis * self.sphere_radius
                sphere_center = spraygun_tip + z_axis * self.sphere_radius
                # xのみ追従、y,zは固定
                sc = np.array([sphere_center[0], fixed_y, fixed_z])
            except Exception:
                sc = self.initial_sphere_center.copy()
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp     = self.get_clock().now().to_msg()
        marker.ns = "sphere_center"
        marker.id = 1000
        marker.type   = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(sc[0])
        marker.pose.position.y = float(sc[1])
        marker.pose.position.z = float(sc[2])
        marker.scale.x = marker.scale.y = marker.scale.z = self.sphere_radius * 2.0
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.2,0.2,1.0,0.8
        self.marker_pub.publish(marker)

    def _pub_exec_path_marker(self):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "exec_path"
        marker.id = 2000
        marker.type   = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.points = self.exec_pts
        marker.scale.x = 0.002
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 0.0,1.0,0.0,1.0
        self.marker_pub.publish(marker)

    def _record_ee_position(self):
        try:
            tf = self.tf_buffer.lookup_transform("base_link","ee_link",rclpy.time.Time())
            ee_p = np.array([tf.transform.translation.x,
                             tf.transform.translation.y,
                             tf.transform.translation.z])
            q = tf.transform.rotation
            z_axis = quaternion_to_rotmat([q.x,q.y,q.z,q.w])[:,2]
            spray_tip = ee_p + z_axis * self.spraygun_length
            pt = Point(x=spray_tip[0], y=spray_tip[1], z=spray_tip[2])
            self.exec_pts.append(pt)
            now = time.time()
            if self.prev_tip_pos is not None:
                dt = now - self.prev_time
                if dt>0:
                    v = math.sqrt((pt.x-self.prev_tip_pos.x)**2 +
                                  (pt.y-self.prev_tip_pos.y)**2 +
                                  (pt.z-self.prev_tip_pos.z)**2)/dt
                    self.csv_writer.writerow([now,pt.x,pt.y,pt.z,v])
                    self.speed_sum += v
                    self.speed_count += 1
            self.prev_tip_pos, self.prev_time = pt, now
        except:
            pass

    def _send_first_point(self):
        if self.idx >= len(self.arc_points):
            self.get_logger().info("全目標点到達完了")
            return self._finish_and_log_avg_speed()

        pose = self.arc_points[self.idx][0]
        for attempt in range(5):
            req_ik = GetPositionIK.Request()
            req_ik.ik_request.group_name = "ar_manipulator"
            req_ik.ik_request.pose_stamped = pose
            req_ik.ik_request.robot_state.joint_state.name = self.joint_names
            seed = (np.array(self.cur_positions) +
                    np.random.uniform(-0.1,0.1,6)).tolist() if attempt>0 else self.cur_positions
            req_ik.ik_request.robot_state.joint_state.position = seed
            req_ik.ik_request.timeout = DurationMsg(sec=0, nanosec=int(0.3e9))
            fut = self.ik_client.call_async(req_ik)
            rclpy.spin_until_future_complete(self, fut)
            res = fut.result()
            if res.error_code.val == res.error_code.SUCCESS:
                break
        else:
            self.get_logger().error(f"[IK失敗 idx={self.idx}]")
            self.idx += 1
            return self._send_first_point()

        target = list(res.solution.joint_state.position)
        jc_list=[]
        for n,p in zip(self.joint_names, target):
            jc = JointConstraint()
            jc.joint_name      = n
            jc.position        = p
            jc.tolerance_above = 0.02
            jc.tolerance_below = 0.02
            jc.weight          = 1.0
            jc_list.append(jc)
        oc = OrientationConstraint()
        oc.link_name                  = "ee_link"
        oc.header.frame_id            = pose.header.frame_id
        oc.orientation                = pose.pose.orientation
        oc.absolute_x_axis_tolerance  = 0.02
        oc.absolute_y_axis_tolerance  = 0.02
        oc.absolute_z_axis_tolerance  = 0.02
        oc.weight                     = 1.0

        c = Constraints()
        c.joint_constraints       = jc_list
        c.orientation_constraints = [oc]

        req = MotionPlanRequest()
        req.group_name                       = "ar_manipulator"
        req.max_velocity_scaling_factor      = self.speed_scaling
        req.max_acceleration_scaling_factor  = self.speed_scaling
        req.goal_constraints.append(c)
        req.start_state.joint_state.name     = self.joint_names
        req.start_state.joint_state.position = self.cur_positions

        goal = MoveGroup.Goal()
        goal.request = req
        self.move_client.send_goal_async(goal).add_done_callback(self._goal_response_cb)

    def _goal_response_cb(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error("初期MoveGroup拒否")
            return self._finish_and_log_avg_speed()
        handle.get_result_async().add_done_callback(self._result_cb_first_point)

    def _result_cb_first_point(self, future):
        res = future.result().result
        if res.error_code.val != 1:
            self.get_logger().error(f"初期実行エラー:{res.error_code.val}")
        self.idx += 1
        self._execute_cartesian_path_from_second()

    def _execute_cartesian_path_from_second(self):
        if self.idx >= len(self.arc_points):
            self.get_logger().info("CartesianPath完了")
            return self._finish_and_log_avg_speed()

        waypoints = [p[0].pose for p in self.arc_points[self.idx:]]
        cli = self.create_client(GetCartesianPath, "compute_cartesian_path")
        cli.wait_for_service()

        req = GetCartesianPath.Request()
        req.group_name                       = "ar_manipulator"
        req.waypoints                        = waypoints
        req.max_step                         = 0.005
        req.jump_threshold                   = 0.0
        req.start_state.joint_state.name     = self.joint_names
        req.start_state.joint_state.position = self.cur_positions

        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res.error_code.val != res.error_code.SUCCESS:
            self.get_logger().error(f"[CartesianPath]{res.error_code.val}")
            return self._finish_and_log_avg_speed()
        if getattr(res, "fraction",1.0) < 0.999:
            self.get_logger().warn(f"到達率低い:{res.fraction:.3f}")

        traj = res.solution
        if self.speed_scaling < 1.0:
            for pt in traj.joint_trajectory.points:
                t = pt.time_from_start.sec + pt.time_from_start.nanosec*1e-9
                nt = t/self.speed_scaling
                sec = int(nt); nan = int((nt-sec)*1e9)
                pt.time_from_start = DurationMsg(sec=sec, nanosec=nan)

        exec_client = ActionClient(self, ExecuteTrajectory, "execute_trajectory")
        exec_client.wait_for_server()
        g = ExecuteTrajectory.Goal()
        g.trajectory = traj
        exec_client.send_goal_async(g).add_done_callback(lambda f: self._finish_and_log_avg_speed())

    def _finish_and_log_avg_speed(self):
        avg = self.speed_sum/self.speed_count if self.speed_count>0 else 0.0
        self.get_logger().info(f"=== 平均速度: {avg:.4f} m/s ===")
        self.csv_file.close()
        rclpy.shutdown()

def main():
    rclpy.init()
    node = MoveOnSphereFollowingX0()
    rclpy.spin(node)

if __name__=="__main__":
    main()

