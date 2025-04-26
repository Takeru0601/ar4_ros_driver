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
from visualization_msgs.msg import Marker  # <-- 追加！

class MoveAlongArc(Node):
    def __init__(self):
        super().__init__('move_along_arc_client')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self._marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)  # <-- 追加！
        self.arc_points = self.generate_arc_points()
        self.current_index = 0

        self.get_logger().info('Waiting for MoveGroup action server...')
        self._action_client.wait_for_server()
        self.send_next_goal()

    def generate_arc_points(self):
        points = []
        radius = 0.05
        center = [0.0, -0.35, 0.35]
        fixed_y = center[1]

        # --- 中心点を表示 ---
        center_marker = Marker()
        center_marker.header.frame_id = 'base_link'
        center_marker.type = Marker.SPHERE
        center_marker.action = Marker.ADD
        center_marker.scale.x = 0.02
        center_marker.scale.y = 0.02
        center_marker.scale.z = 0.02
        center_marker.color.r = 1.0
        center_marker.color.g = 0.0
        center_marker.color.b = 0.0
        center_marker.color.a = 1.0
        center_marker.pose.position.x = center[0]
        center_marker.pose.position.y = center[1]
        center_marker.pose.position.z = center[2]
        center_marker.pose.orientation.w = 1.0
        center_marker.id = 0
        self._marker_pub.publish(center_marker)

        # --- 軌道ラインを準備 ---
        line_marker = Marker()
        line_marker.header.frame_id = 'base_link'
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.005  # 線の太さ
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0
        line_marker.pose.orientation.w = 1.0
        line_marker.id = 1

        start_angle = math.radians(0)
        end_angle = math.radians(90)
        steps = 5

        for i in range(steps + 1):
            theta = start_angle + (end_angle - start_angle) * i / steps
            x = center[0] + radius * math.cos(theta)
            z = center[2] + radius * math.sin(theta)
            y = fixed_y

            dir_x = center[0] - x
            dir_y = center[1] - y
            dir_z = center[2] - z
            norm = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
            dir_x /= norm
            dir_y /= norm
            dir_z /= norm

            up = [0, 1, 0]
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
            line_marker.points.append(Point(x=x, y=y, z=z))  # 軌道ラインに追加

        self._marker_pub.publish(line_marker)

        return points

    def send_next_goal(self):
        if self.current_index >= len(self.arc_points):
            self.get_logger().info('✅ All arc points executed!')
            rclpy.shutdown()
            return

        pose = self.arc_points[self.current_index]
        self.get_logger().info(f'▶️ Sending point {self.current_index + 1}/{len(self.arc_points)}')

        # --- EEのz軸を可視化 ---
        z_axis_marker = Marker()
        z_axis_marker.header.frame_id = 'base_link'
        z_axis_marker.type = Marker.ARROW
        z_axis_marker.action = Marker.ADD
        z_axis_marker.scale.x = 0.03  # 矢印の長さ
        z_axis_marker.scale.y = 0.005
        z_axis_marker.scale.z = 0.005
        z_axis_marker.color.r = 0.0
        z_axis_marker.color.g = 0.0
        z_axis_marker.color.b = 1.0
        z_axis_marker.color.a = 1.0
        z_axis_marker.pose = pose.pose
        z_axis_marker.id = 100 + self.current_index
        self._marker_pub.publish(z_axis_marker)

        # --- MoveGroup Goal 送信 ---
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

def main(args=None):
    rclpy.init(args=args)
    node = MoveAlongArc()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
