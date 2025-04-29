import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
import tf_transformations


class ZAxisVisualizer(Node):
    def __init__(self):
        super().__init__('z_axis_visualizer')
        self.marker_pub = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)
        self.timer = self.create_timer(0.5, self.publish_arc_with_z_arrows)

        # 中心点と半径
        self.center = np.array([0.0, -0.35, 0.35])
        self.radius = 0.2
        self.inner_exclusion_radius = 0.1
        self.num_points = 40

    def publish_arc_with_z_arrows(self):
        marker_array = MarkerArray()

        # 円の点生成
        for i in range(self.num_points):
            angle = 2 * np.pi * i / self.num_points
            x = self.center[0] + self.radius * np.cos(angle)
            y = self.center[1] + self.radius * np.sin(angle)
            z = self.center[2]

            point = np.array([x, y, z])
            direction = self.center - point
            distance = np.linalg.norm(direction)

            if distance < self.inner_exclusion_radius:
                continue  # 中心点から近すぎる点はスキップ

            direction /= distance  # 単位ベクトル

            # クォータニオン計算
            quat = self.z_axis_to_quaternion(direction)

            marker = Marker()
            marker.header.frame_id = "base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "z_axis_arrows"
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
            marker.scale.x = 0.05
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            marker_array.markers.append(marker)

        # 半透明の中心球を表示
        center_marker = Marker()
        center_marker.header.frame_id = "base_link"
        center_marker.header.stamp = self.get_clock().now().to_msg()
        center_marker.ns = "center_sphere"
        center_marker.id = 999
        center_marker.type = Marker.SPHERE
        center_marker.action = Marker.ADD
        center_marker.pose.position.x = self.center[0]
        center_marker.pose.position.y = self.center[1]
        center_marker.pose.position.z = self.center[2]
        center_marker.scale.x = self.inner_exclusion_radius * 2
        center_marker.scale.y = self.inner_exclusion_radius * 2
        center_marker.scale.z = self.inner_exclusion_radius * 2
        center_marker.color.r = 1.0
        center_marker.color.g = 0.0
        center_marker.color.b = 0.0
        center_marker.color.a = 0.3  # 半透明

        marker_array.markers.append(center_marker)

        self.marker_pub.publish(marker_array)

    def z_axis_to_quaternion(self, z_dir):
        z_axis = np.array(z_dir)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Z軸に直交する任意のベクトルを選ぶ（例: X軸に近いが非平行）
        arbitrary = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.99 else np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(arbitrary, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)

        rot_matrix = np.array([
            [x_axis[0], y_axis[0], z_axis[0], 0],
            [x_axis[1], y_axis[1], z_axis[1], 0],
            [x_axis[2], y_axis[2], z_axis[2], 0],
            [0.0,       0.0,       0.0,       1.0]
        ])

        return tf_transformations.quaternion_from_matrix(rot_matrix)


def main():
    rclpy.init()
    node = ZAxisVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
