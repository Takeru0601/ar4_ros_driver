#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
import tf_transformations
import random
import math

class PointCloudVisualizer(Node):
    def __init__(self):
        super().__init__('point_cloud_visualizer')

        # === パラメータ ===
        self.center = [0.0, -0.35, 0.35]
        self.radius_threshold = 0.1  # 中心からこの距離未満の点は削除
        self.num_points = 40         # 矢印の数

        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)

        # 可視化マーカーを送信
        self.publish_center_marker()
        self.publish_inner_sphere_marker()
        self.publish_point_arrows()

    def generate_valid_random_points(self):
        points = []
        while len(points) < self.num_points:
            x = random.uniform(-0.2, 0.2)
            y = random.uniform(-0.6, -0.1)
            z = random.uniform(0.1, 0.6)

            dx = x - self.center[0]
            dy = y - self.center[1]
            dz = z - self.center[2]
            dist = math.sqrt(dx**2 + dy**2 + dz**2)

            if dist >= self.radius_threshold:
                points.append((x, y, z))
        return points

    def compute_orientation_quat(self, from_point, to_point):
        dir_x = to_point[0] - from_point[0]
        dir_y = to_point[1] - from_point[1]
        dir_z = to_point[2] - from_point[2]
        norm = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        dir_x /= norm
        dir_y /= norm
        dir_z /= norm

        # Y軸を up ベクトルとして姿勢を定義（Z軸が中心を向く）
        up = [0, 1, 0]
        x_axis = [
            up[1]*dir_z - up[2]*dir_y,
            up[2]*dir_x - up[0]*dir_z,
            up[0]*dir_y - up[1]*dir_x,
        ]
        x_norm = math.sqrt(sum(v**2 for v in x_axis))
        x_axis = [v / x_norm for v in x_axis]

        y_axis = [
            dir_y*x_axis[2] - dir_z*x_axis[1],
            dir_z*x_axis[0] - dir_x*x_axis[2],
            dir_x*x_axis[1] - dir_y*x_axis[0],
        ]

        rot_matrix = [
            [x_axis[0], y_axis[0], dir_x],
            [x_axis[1], y_axis[1], dir_y],
            [x_axis[2], y_axis[2], dir_z],
        ]
        quat = tf_transformations.quaternion_from_matrix([
            [rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], 0],
            [rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], 0],
            [rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], 0],
            [0, 0, 0, 1]
        ])
        return quat

    def publish_point_arrows(self):
        points = self.generate_valid_random_points()
        for i, (x, y, z) in enumerate(points):
            quat = self.compute_orientation_quat((x, y, z), self.center)

            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'ee_z_axis'
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.01
            marker.scale.y = 0.015
            marker.scale.z = 0.1
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            start = Point(x=x, y=y, z=z)

            rot_matrix = tf_transformations.quaternion_matrix(quat)
            z_axis = [rot_matrix[0][2], rot_matrix[1][2], rot_matrix[2][2]]
            arrow_length = 0.05

            end = Point()
            end.x = x + arrow_length * z_axis[0]
            end.y = y + arrow_length * z_axis[1]
            end.z = z + arrow_length * z_axis[2]

            marker.points.append(start)
            marker.points.append(end)

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

    def publish_inner_sphere_marker(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'radius_sphere'
        marker.id = 1001
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.center[0]
        marker.pose.position.y = self.center[1]
        marker.pose.position.z = self.center[2]
        marker.scale.x = self.radius_threshold * 2
        marker.scale.y = self.radius_threshold * 2
        marker.scale.z = self.radius_threshold * 2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.25  # 半透明
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
