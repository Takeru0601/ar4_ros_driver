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


class MoveAlongArc(Node):
    def __init__(self):
        super().__init__('move_along_arc_client')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')

        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)

        self.arc_points = self.generate_arc_points()
        self.current_index = 0

        self.get_logger().info('Waiting for MoveGroup action server...')
        self._action_client.wait_for_server()

        self.publish_center_marker()
        self.publish_arc_line_marker()

        self.send_next_goal()

    def generate_arc_points(self):
        points = []
        self.radius = 0.1
        self.center = [0.0, -0.35, 0.35]
        fixed_y = self.center[1]

        start_angle = math.radians(0)
        end_angle = math.radians(180)
        steps = 10

        for i in range(steps + 1):
            theta = start_angle + (end_angle - start_angle) * i / steps
            x = self.center[0] + self.radius * math.cos(theta)
            z = self.center[2] + self.radius * math.sin(theta)
            y = fixed_y

            # -------- 姿勢計算（Z軸を中心向きに合わせる） --------
            z_axis = [
                self.center[0] - x,
                self.center[1] - y,
                self.center[2] - z
            ]
            z_norm = math.sqrt(sum(c**2 for c in z_axis))
            z_axis = [c / z_norm for c in z_axis]

            # X軸はできるだけ水平にとる（ここでは世界Y軸との外積）
            world_up = [0.0, 1.0, 0.0]
            x_axis = [
                world_up[1]*z_axis[2] - world_up[2]*z_axis[1],
                world_up[2]*z_axis[0] - world_up[0]*z_axis[2],
                world_up[0]*z_axis[1] - world_up[1]*z_axis[0]
            ]
            x_norm = math.sqrt(sum(c**2 for c in x_axis))
            if x_norm < 1e-6:
                x_axis = [1.0, 0.0, 0.0]  # 特殊ケース
            else:
                x_axis = [c / x_norm for c in x_axis]

            # Y軸は直交させる
            y_axis = [
                z_axis[1]*x_axis[2] - z_axis[2]*x_axis[1],
                z_axis[2]*x_axis[0] - z_axis[0]*x_axis[2],
                z_axis[0]*x_axis[1] - z_axis[1]*x_axis[0]
            ]

            rot_matrix = [
                [x_axis[0], y_axis[0], z_axis[0], 0],
                [x_axis[1], y_axis[1], z_axis[1], 0],
                [x_axis[2], y_axis[2], z_axis[2], 0],
                [0, 0, 0, 1]
            ]

            quat = tf_transformations.quaternion_from_matrix(rot_matrix)
            # ---------------------------------------------

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

        self.publish_ee_z_axis(pose, self.current_index)

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = 'ar_manipulator'
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3

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

    def publish_ee_z_axis(self, pose, id_num):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'ee_z_axis'
        marker.id = id_num
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.scale.x = 0.05
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)

    def publish_center_marker(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'arc_center'
        marker.id = 1000
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
        marker.ns = 'arc_path'
        marker.id = 2000
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.005
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        for pose in self.arc_points:
            point = Point()
            point.x = pose.pose.position.x
            point.y = pose.pose.position.y
            point.z = pose.pose.position.z
            marker.points.append(point)
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = MoveAlongArc()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
