#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Duration
import math

class ZAxisVisualizer(Node):
    def __init__(self):
        super().__init__('z_axis_visualizer')
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)

        self.center = [0.0, -0.35, 0.35]
        self.radius = 0.1  # 半径30mm
        self.steps = 100

        self.create_timer(1.0, self.publish_markers)

    def publish_markers(self):
        self.publish_center_sphere()
        self.publish_point_cloud()

    def publish_center_sphere(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'center'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.center[0]
        marker.pose.position.y = self.center[1]
        marker.pose.position.z = self.center[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = self.radius * 2
        marker.scale.y = self.radius * 2
        marker.scale.z = self.radius * 2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.4
        marker.lifetime = Duration()  # 無限表示
        self.marker_pub.publish(marker)

    def publish_point_cloud(self):
        marker_id = 1
        for i in range(self.steps):
            theta = 2 * math.pi * i / self.steps
            phi = math.acos(2 * i / self.steps - 1)

            x = 0.2 * math.sin(phi) * math.cos(theta) + self.center[0]
            y = 0.2 * math.sin(phi) * math.sin(theta) + self.center[1]
            z = 0.2 * math.cos(phi) + self.center[2]

            dist = math.sqrt((x - self.center[0])**2 + (y - self.center[1])**2 + (z - self.center[2])**2)
            if dist < 0.05:　#一定距離内の矢印は削除
                continue

            self.publish_arrow(x, y, z, marker_id)
            marker_id += 1

    def publish_arrow(self, x, y, z, marker_id):
        start = Point(x=x, y=y, z=z)
        direction = [
            self.center[0] - x,
            self.center[1] - y,
            self.center[2] - z
        ]
        norm = math.sqrt(sum(d ** 2 for d in direction))
        direction = [d / norm for d in direction]
        arrow_length = 0.05

        end = Point()
        end.x = x + direction[0] * arrow_length
        end.y = y + direction[1] * arrow_length
        end.z = z + direction[2] * arrow_length

        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'z_axis_arrows'
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.01
        marker.scale.y = 0.015
        marker.scale.z = 0.02
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.points = [start, end]
        marker.lifetime = Duration()  # 無限表示
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = ZAxisVisualizer()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
