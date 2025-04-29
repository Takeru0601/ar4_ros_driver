#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import tf_transformations
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker

class ZAxisVisualizer(Node):
    def __init__(self):
        super().__init__('z_axis_visualizer')
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)

        self.center = [0.0, -0.35, 0.35]
        self.radius = 0.15  # 曲率半径
        self.point_count = 40

        self.publish_center_sphere()
        self.publish_arc_with_z_arrows()

    def publish_arc_with_z_arrows(self):
        for i in range(self.point_count):
            angle = 2 * math.pi * i / self.point_count
            x = self.center[0] + self.radius * math.cos(angle)
            z = self.center[2] + self.radius * math.sin(angle)
            y = self.center[1]

            # 中心点との距離
            dist = math.sqrt((x - self.center[0])**2 + (y - self.center[1])**2 + (z - self.center[2])**2)
            if dist < 0.1:
                continue  # 半径0.1m以内の点はスキップ

            # EEのz軸を中心点に向ける向き（dirベクトル）
            dir_x = self.center[0] - x
            dir_y = self.center[1] - y
            dir_z = self.center[2] - z
            norm = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
            dir_x /= norm
            dir_y /= norm
            dir_z /= norm

            # 基本姿勢のZ軸がdirになるような回転
            quat = tf_transformations.quaternion_from_matrix([
                [0, 0, dir_x, 0],
                [0, 1, dir_y, 0],
                [-1, 0, dir_z, 0],
                [0, 0, 0, 1],
            ])

            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'z_axis_arrows'
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            marker.pose.orientation.x = quat[0]
            marker.pose.orientation.y = quat[1]
            marker.pose.orientation.z = quat[2]
            marker.pose.orientation.w = quat[3]
            marker.scale.x = 0.05  # 矢印の長さ
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
            self.marker_pub.publish(marker)

    def publish_center_sphere(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'center_sphere'
        marker.id = 999
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.center[0]
        marker.pose.position.y = self.center[1]
        marker.pose.position.z = self.center[2]
        marker.scale.x = 0.2  # 直径0.2m = 半径0.1m
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.2  # 半透明
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = ZAxisVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
