#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import math
import tf_transformations
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint
from shape_msgs.msg import SolidPrimitive


class MoveAlongArc(Node):
    def __init__(self):
        super().__init__('move_along_arc_client')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.arc_points = self.generate_arc_points()
        self.current_index = 0

        self.get_logger().info('Waiting for MoveGroup action server...')
        self._action_client.wait_for_server()
        self.send_next_goal()

    def generate_arc_points(self):
        points = []
        radius = 0.05  # 半径
        center = [0.0, -0.35, 0.35]  # 円弧の中心
        fixed_y = center[1]

        start_angle = math.radians(0)
        end_angle = math.radians(90)
        steps = 5

        for i in range(steps + 1):
            theta = start_angle + (end_angle - start_angle) * i / steps
            x = center[0] + radius * math.cos(theta)
            z = center[2] + radius * math.sin(theta)
            y = fixed_y

            # 中心へのベクトル（z軸をこの方向に向けたい）
            dir_x = center[0] - x
            dir_y = center[1] - y
            dir_z = center[2] - z
            norm = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
            dir_x /= norm
            dir_y /= norm
            dir_z /= norm

            # z軸を dir に向ける回転行列を作成
            # x軸の候補（Y軸ベース）
            up = [0, 1, 0]
            x_axis = [
                up[1]*dir_z - up[2]*dir_y,
                up[2]*dir_x - up[0]*dir_z,
                up[0]*dir_y - up[1]*dir_x,
            ]
            # x軸を正規化
            x_norm = math.sqrt(x_axis[0]**2 + x_axis[1]**2 + x_axis[2]**2)
            x_axis = [v / x_norm for v in x_axis]

            # y軸を再定義（z × x）
            new_y = [
                dir_y*x_axis[2] - dir_z*x_axis[1],
                dir_z*x_axis[0] - dir_x*x_axis[2],
                dir_x*x_axis[1] - dir_y*x_axis[0],
            ]

            # 回転行列からクォータニオン生成
            rot_matrix = [
                [x_axis[0], new_y[0], dir_x],
                [x_axis[1], new_y[1], dir_y],
                [x_axis[2], new_y[2], dir_z],
            ]
            quat = tf_transformations.quaternion_from_matrix([
                [rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], 0],
                [rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], 0],
                [rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], 0],
                [0, 0, 0, 1]
            ])

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
            self.get_logger().info('✅ All arc points executed!')
            rclpy.shutdown()
            return

        pose = self.arc_points[self.current_index]
        self.get_logger().info(f'▶️ Sending point {self.current_index + 1}/{len(self.arc_points)}')

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = 'ar_manipulator'
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3

        # --- Position Constraint ---
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

        goal_constraints = Constraints()
        goal_constraints.position_constraints.append(position_constraint)
        # 姿勢制約なし（姿勢はposeに反映済み）
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


def main(args=None):
    rclpy.init(args=args)
    node = MoveAlongArc()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
