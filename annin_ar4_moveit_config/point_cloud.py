#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
import tf_transformations
import numpy as np
import math

from moveit_commander.robot_trajectory import RobotTrajectory
from moveit_commander import PlanningSceneInterface, RobotCommander
import moveit_commander

class ReachableArrowVisualizer(Node):
    def __init__(self):
        super().__init__('reachable_arrow_visualizer')

        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)

        # パラメータ
        self.center = np.array([0.0, -0.35, 0.35])
        self.radius_threshold = 0.1
        self.num_points = 300

        # MoveIt 初期化
        moveit_commander.roscpp_initialize([])
        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander('ar_manipulator')

        self.get_logger().info('⚙️ Calculating reachable directions...')
        self.visualize()

    def visualize(self):
        points = self.generate_points_on_sphere(self.num_points)
        reachable = 0
        marker_id = 0

        self.publish_transparent_sphere()

        for point in points:
            direction = self.center - point
            direction /= np.linalg.norm(direction)

            pose = PoseStamped()
            pose.header.frame_id = 'base_link'
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = float(point[2])

            quat = self.direction_to_quaternion(direction)
            pose.pose.orientation.x = quat[0]
            pose.pose.orientation.y = quat[1]
            pose.pose.orientation.z = quat[2]
            pose.pose.orientation.w = quat[3]

            # 半径内ならスキップ
            if np.linalg.norm(point - self.center) < self.radius_threshold:
                continue

            # IK確認
            self.group.set_pose_target(pose)
            plan = self.group.plan()
            if not plan or not plan[0].joint_trajectory.points:
                continue

            reachable += 1
            self.publish_arrow(pose, marker_id)
            marker_id += 1

        self.get_logger().info(f'✅ Done. Reachable: {reachable} / {self.num_points}')

    def direction_to_quaternion(self, direction):
        z_axis = direction
        up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(up, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        rot_matrix = np.identity(4)
        rot_matrix[0:3, 0] = x_axis
        rot_matrix[0:3, 1] = y_axis
        rot_matrix[0:3, 2] = z_axis

        quat = tf_transformations.quaternion_from_matrix(rot_matrix)
        return quat

    def generate_points_on_sphere(self, n):
        points = []
        offset = 2.0 / n
        increment = math.pi * (3.0 - math.sqrt(5.0))  # Golden angle

        for i in range(n):
            y = ((i * offset) - 1) + (offset / 2)
            r = math.sqrt(1 - y * y)
            phi = i * increment
            x = math.cos(phi) * r
            z = math.sin(phi) * r

            point = self.center + 0.15 * np.array([x, y, z])  # 半径0.15の球面上
            points.append(point)
        return points

    def publish_arrow(self, pose, marker_id):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'ee_arrows'
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.scale.y = 0.015
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        quat = (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w,
        )
        rot_matrix = tf_transformations.quaternion_matrix(quat)
        z_axis = rot_matrix[0:3, 2]

        # 終点（エンドエフェクタ位置）
        end = Point()
        end.x = pose.pose.position.x
        end.y = pose.pose.position.y
        end.z = pose.pose.position.z

        # 始点：Z軸方向に後ろへ 5cm
        start = Point()
        start.x = end.x - 0.05 * z_axis[0]
        start.y = end.y - 0.05 * z_axis[1]
        start.z = end.z - 0.05 * z_axis[2]

        marker.points.append(start)
        marker.points.append(end)

        self.marker_pub.publish(marker)

    def publish_transparent_sphere(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'sphere'
        marker.id = 9999
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(self.center[0])
        marker.pose.position.y = float(self.center[1])
        marker.pose.position.z = float(self.center[2])
        marker.scale.x = self.radius_threshold * 2
        marker.scale.y = self.radius_threshold * 2
        marker.scale.z = self.radius_threshold * 2
        marker.color.r = 0.8
        marker.color.g = 0.8
        marker.color.b = 0.8
        marker.color.a = 0.3  # 半透明
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = ReachableArrowVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
