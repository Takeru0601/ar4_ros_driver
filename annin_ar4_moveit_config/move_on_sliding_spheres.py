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

class MoveOnSlidingSphere(Node):
    def __init__(self):
        super().__init__('move_on_sliding_sphere')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)

        # 球の初期中心座標とパラメータ
        self.center_base = [0.0, -0.45, 0.20]
        self.radius = 0.8  # 直径800mm
        self.slide_step = 0.01

        # 平面群 (Y軸方向固定)
        self.y_planes = [-0.45, -0.43, -0.41, -0.39, -0.37, -0.35]
        self.plane_normal = [0.0, 1.0, 0.0]

        # 状態
        self.center_x = self.center_base[0]
        self.current_index = 0
        self.trajectory = []

        self.get_logger().info('Waiting for MoveGroup action server...')
        self._action_client.wait_for_server()

        self.update_arc_points()
        self.publish_center_marker()
        self.publish_arc_line_marker()

        self.send_next_goal()

    def update_arc_points(self):
        self.arc_points = []
        steps = 18
        for y in self.y_planes:
            dy = y - self.center_base[1]
            if abs(dy) > self.radius:
                continue
            circle_radius = math.sqrt(self.radius**2 - dy**2)
            for i in range(steps + 1):
                theta = math.radians(180 * i / steps)
                x = self.center_x + circle_radius * math.cos(theta)
                z = self.center_base[2] + circle_radius * math.sin(theta)

                # 球中心への方向ベクトル
                dir_vec = [
                    self.center_x - x,
                    self.center_base[1] - y,
                    self.center_base[2] - z
                ]
                norm = math.sqrt(sum(v**2 for v in dir_vec))
                dir_vec = [v / norm for v in dir_vec]

                # Z軸をdir_vecに揃えるクォータニオン
                quat = tf_transformations.quaternion_from_matrix(
                    tf_transformations.rotation_matrix_from_vectors([0, 0, 1], dir_vec)
                )

                pose = PoseStamped()
                pose.header.frame_id = 'base_link'
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = z
                pose.pose.orientation.x = quat[0]
                pose.pose.orientation.y = quat[1]
                pose.pose.orientation.z = quat[2]
                pose.pose.orientation.w = quat[3]

                self.arc_points.append(pose)

    def send_next_goal(self):
        if self.current_index >= len(self.arc_points):
            self.center_x += self.slide_step
            self.get_logger().info(f'➡️ Sphere moved to x={self.center_x:.3f}')
            self.current_index = 0
            self.update_arc_points()
            self.publish_center_marker()
            self.publish_arc_line_marker()
            if self.center_x > self.center_base[0] + 0.05:
                self.get_logger().info('✅ All motions completed.')
                rclpy.shutdown()
                return

        pose = self.arc_points[self.current_index]
        self.get_logger().info(f'▶️ Sending point {self.current_index + 1}/{len(self.arc_points)}')
        self.publish_ee_z_axis(pose, self.current_index)
        self.publish_trajectory_marker(pose)

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
        self.current_index += 1
        self.send_next_goal()

    def publish_center_marker(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'sphere_center'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.center_x
        marker.pose.position.y = self.center_base[1]
        marker.pose.position.z = self.center_base[2]
        marker.scale.x = self.radius * 2
        marker.scale.y = self.radius * 2
        marker.scale.z = self.radius * 2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.3
        self.marker_pub.publish(marker)

    def publish_arc_line_marker(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'arc_path'
        marker.id = 1
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.003
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.points = [Point(x=p.pose.position.x, y=p.pose.position.y, z=p.pose.position.z) for p in self.arc_points]
        self.marker_pub.publish(marker)

    def publish_ee_z_axis(self, pose, id_num):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'ee_z'
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

        start = pose.pose.position
        quat = (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        )
        rot = tf_transformations.quaternion_matrix(quat)
        z_axis = [rot[0][2], rot[1][2], rot[2][2]]

        end = Point()
        end.x = start.x + 0.05 * z_axis[0]
        end.y = start.y + 0.05 * z_axis[1]
        end.z = start.z + 0.05 * z_axis[2]

        marker.points = [start, end]
        self.marker_pub.publish(marker)

    def publish_trajectory_marker(self, pose):
        self.trajectory.append(Point(
            x=pose.pose.position.x,
            y=pose.pose.position.y,
            z=pose.pose.position.z
        ))
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'ee_trajectory'
        marker.id = 2
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.002
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.points = self.trajectory
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = MoveOnSlidingSphere()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
