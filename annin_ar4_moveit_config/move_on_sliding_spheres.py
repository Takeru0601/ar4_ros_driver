#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import math
import tf_transformations
from geometry_msgs.msg import PoseStamped, Point
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker


class MoveOnSphereIntersections(Node):
    def __init__(self):
        super().__init__('move_on_sphere_intersections')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)

        # 球の中心と半径 (直径0.15m -> 半径0.075m)
        self.center = [0.0, -0.45, 0.20]
        self.radius = 0.075

        # 塗装断面となる平面群 (y = 定数)
        self.y_planes = [-0.45, -0.43, -0.41, -0.39, -0.37, -0.35]
        # 各平面の法線ベクトル (すべて y 軸方向)
        self.plane_normals = [[0.0, 1.0, 0.0] for _ in self.y_planes]

        # 交線上の点と姿勢を生成
        self.arc_points = self.generate_intersection_points()
        self.current_index = 0

        self.get_logger().info('Waiting for MoveGroup action server...')
        self._action_client.wait_for_server()

        self.publish_center_marker()
        self.publish_arc_line_marker()

        self.send_next_goal()

    def generate_intersection_points(self):
        points = []
        steps = 18  # 180° / 5°

        for plane_idx, y in enumerate(self.y_planes):
            normal = self.plane_normals[plane_idx]
            dy = y - self.center[1]
            if abs(dy) > self.radius:
                continue
            circle_radius = math.sqrt(self.radius**2 - dy**2)

            for i in range(steps + 1):
                # 左右ジグザグ走査
                if plane_idx % 2 == 0:
                    theta = math.radians(180 * i / steps)
                else:
                    theta = math.radians(180 * (steps - i) / steps)

                # 球面断面の座標
                x = self.center[0] + circle_radius * math.cos(theta)
                z = self.center[2] + circle_radius * math.sin(theta)

                # 1) 球の中心へのベクトルを計算
                dir_x = self.center[0] - x
                dir_y = self.center[1] - y
                dir_z = self.center[2] - z
                norm = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
                dir_x /= norm
                dir_y /= norm
                dir_z /= norm

                # 2) 平面法線との内積を引いて射影
                dot = dir_x * normal[0] + dir_y * normal[1] + dir_z * normal[2]
                dir_x -= dot * normal[0]
                dir_y -= dot * normal[1]
                dir_z -= dot * normal[2]
                # 正規化
                norm2 = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
                dir_x /= norm2
                dir_y /= norm2
                dir_z /= norm2

                # 3) 法線を平面内のベクトルとして扱い、直行基底を作成
                up = [0.0, 1.0, 0.0]
                x_axis = [
                    up[1]*dir_z - up[2]*dir_y,
                    up[2]*dir_x - up[0]*dir_z,
                    up[0]*dir_y - up[1]*dir_x,
                ]
                x_norm = math.sqrt(sum(v**2 for v in x_axis))
                x_axis = [v / x_norm for v in x_axis]
                new_y = [
                    dir_y*x_axis[2] - dir_z*x_axis[1],
                    dir_z*x_axis[0] - dir_x*x_axis[2],
                    dir_x*x_axis[1] - dir_y*x_axis[0],
                ]

                rot_matrix = [
                    [x_axis[0], new_y[0], dir_x],
                    [x_axis[1], new_y[1], dir_y],
                    [x_axis[2], new_y[2], dir_z],
                ]
                quat = tf_transformations.quaternion_from_matrix([
                    [rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], 0.0],
                    [rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], 0.0],
                    [rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                ])

                # PoseStamped にまとめる
                pose = PoseStamped()
                pose.header.frame_id = 'base_link'
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = z
                pose.pose.orientation.x = quat[0]
                pose.pose.orientation.y = quat[1]
                pose.pose.orientation.z = quat[2]
                pose.pose.orientation.w = quat[3]
                points.append(pose)
        return points

    def send_next_goal(self):
        if self.current_index >= len(self.arc_points):
            self.get_logger().info('✅ All intersection points executed!')
            rclpy.shutdown()
            return

        pose = self.arc_points[self.current_index]
        self.get_logger().info(f'▶️ Sending point {self.current_index + 1}/{len(self.arc_points)}')
        self.publish_ee_z_axis(pose, self.current_index)

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = 'ar_manipulator'
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3

        # 位置制約
        position_constraint = PositionConstraint()
        position_constraint.header.frame_id = pose.header.frame_id
        position_constraint.link_name = 'ee_link'
        position_constraint.target_point_offset.x = 0.0
        position_constraint.target_point_offset.y = 0.0
        position_constraint.target_point_offset.z = 0.0

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.01, 0.01, 0.01]
        position_constraint.constraint_region.primitives.append(box)
        position_constraint.constraint_region.primitive_poses.append(pose.pose)

        # 姿勢制約
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = pose.header.frame_id
        orientation_constraint.link_name = 'ee_link'
        orientation_constraint.orientation = pose.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.01
        orientation_constraint.absolute_y_axis_tolerance = 0.01
        orientation_constraint.absolute_z_axis_tolerance = 0.01
        orientation_constraint.weight = 1.0

        # ゴールに追加
        goal_constraints = Constraints()
        goal_constraints.position_constraints.append(position_constraint)
        goal_constraints.orientation_constraints.append(orientation_constraint)
        req.goal_constraints.append(goal_constraints)
        goal_msg.request = req

        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ Goal rejected')
            rclpy.shutdown()
            return
        self.get_logger().info('✅ Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'🎯 Result received: {result.error_code}')
        self.current_index += 1
        self.send_next_goal()

    # --- 可視化用関数 ---
    def publish_ee_z_axis(self, pose, id_num):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'ee_z_axis'
        marker.id = id_num
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.scale.y = 0.015
        marker.scale.z = 0.1
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        start = Point()
        start.x = pose.pose.position.x
        start.y = pose.pose.position.y
        start.z = pose.pose.position.z
        marker.points.append(start)

        # Z軸方向ベクトル(エンドエフェクタ姿勢のZ軸)
        # クォータニオンからZ軸ベクトルを計算
        import numpy as np
        q = pose.pose.orientation
        R = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        z_axis = Point()
        z_axis.x = start.x + R[2, 0] * 0.1
        z_axis.y = start.y + R[2, 1] * 0.1
        z_axis.z = start.z + R[2, 2] * 0.1
        marker.points.append(z_axis)

        self.marker_pub.publish(marker)

    def publish_center_marker(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'sphere_center'
        marker.id = 9999
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.center[0]
        marker.pose.position.y = self.center[1]
        marker.pose.position.z = self.center[2]
        marker.scale.x = 0.02
        marker.scale.y = 0.02
        marker.scale.z = 0.02
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)

    def publish_arc_line_marker(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'arc_line'
        marker.id = 10000
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.005
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        for pose in self.arc_points:
            p = Point()
            p.x = pose.pose.position.x
            p.y = pose.pose.position.y
            p.z = pose.pose.position.z
            marker.points.append(p)

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = MoveOnSphereIntersections()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
