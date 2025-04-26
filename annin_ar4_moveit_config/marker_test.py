#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker

class SimpleMarkerPublisher(Node):
    def __init__(self):
        super().__init__('simple_marker_publisher')
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('✅ Marker publisher initialized')

    def timer_callback(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'test_marker'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = 0.0
        marker.pose.position.y = 0.0
        marker.pose.position.z = 0.5
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.1
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)
        self.get_logger().info('📢 Marker published!')

def main(args=None):
    rclpy.init(args=args)
    node = SimpleMarkerPublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
