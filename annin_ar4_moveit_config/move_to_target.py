#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import math
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint
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
        radius = 0.1  # 10cm
        center_y = 0.2
        center_z = 0.2
        x = 0.3  # yz平面上で動くので x は固定

        start_angle = math.radians(0)
        end_angle = math.radians(90)
        steps = 10  # 10点に分割

        for i in range(steps + 1):
            theta = start_angle + (end_angle - start_angle) * i / steps
            y = center_y + radius * math.cos(theta)
            z = center_z + radius * math.sin(theta)

            pose = PoseStamped()
            pose.header.frame_id = 'base_link'
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z

            # 姿勢は固定（真上を向いたまま）
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            pose.pose.orientation.w = 1.0

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
        req.max_velocity_scaling_factor = 0.2
        req.max_acceleration_scaling_factor = 0.2

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

        # --- Orientation Constraint ---
        orientation_constraint = OrientationConstraint()
        orientation_constraint.header.frame_id = pose.header.frame_id
        orientation_constraint.link_name = 'ee_link'
        orientation_constraint.orientation = pose.pose.orientation
        orientation_constraint.absolute_x_axis_tolerance = 0.1
        orientation_constraint.absolute_y_axis_tolerance = 0.1
        orientation_constraint.absolute_z_axis_tolerance = 0.1
        orientation_constraint.weight = 1.0

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

def main(args=None):
    rclpy.init(args=args)
    node = MoveAlongArc()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
