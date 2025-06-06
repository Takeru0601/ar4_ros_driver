#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import tf_transformations
from geometry_msgs.msg import PoseStamped, PoseArray, Pose
from visualization_msgs.msg import Marker, MarkerArray

class MoveOnSlidingSpheres(Node):
    def __init__(self):
        super().__init__('move_on_sliding_spheres')
        self.radius = 0.075  # 直径150mmの球
        self.base_center = [0.4, 0.0, 0.35]  # 初期の球の中心
        self.num_spheres = 10
        self.x_start = -0.05
        self.x_step = 0.01
        self.y_planes = [0.05, 0.07]
        self.plane_normals = [[0.0, 1.0, 0.0]] * len(self.y_planes)
        self.steps_per_circle = 18

        self.pose_pub = self.create_publisher(PoseArray, 'pose_array', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)

        self.timer = self.create_timer(1.0, self.publish)

    def publish(self):
        pose_array = PoseArray()
        pose_array.header.frame_id = 'base_link'
        pose_array.header.stamp = self.get_clock().now().to_msg()

        marker_array = MarkerArray()
        marker_id = 0

        for shift_index in range(self.num_spheres):
            x_shift = self.x_start + shift_index * self.x_step
            center = [
                self.base_center[0] + x_shift,
                self.base_center[1],
                self.base_center[2]
            ]

            # 中心マーカー
            marker = Marker()
            marker.header.frame_id = 'base_link'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'sphere_centers'
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = center[0]
            marker.pose.position.y = center[1]
            marker.pose.position.z = center[2]
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker_array.markers.append(marker)

            for plane_idx, y in enumerate(self.y_planes):
                normal = self.plane_normals[plane_idx]
                dy = y - center[1]
                if abs(dy) > self.radius:
                    continue
                circle_radius = math.sqrt(self.radius**2 - dy**2)

                for i in range(self.steps_per_circle + 1):
                    # ジグザグ順
                    if (shift_index + plane_idx) % 2 == 0:
                        theta = math.radians(180 * i / self.steps_per_circle)
                    else:
                        theta = math.radians(180 * (self.steps_per_circle - i) / self.steps_per_circle)

                    x = center[0] + circle_radius * math.cos(theta)
                    z = center[2] + circle_radius * math.sin(theta)

                    # EEが球の中心を向くように方向ベクトルを計算
                    dir_x = center[0] - x
                    dir_y = center[1] - y
                    dir_z = center[2] - z
                    norm = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
                    dir_x /= norm
                    dir_y /= norm
                    dir_z /= norm

                    # 平面への射影
                    dot = dir_x * normal[0] + dir_y * normal[1] + dir_z * normal[2]
                    dir_x -= dot * normal[0]
                    dir_y -= dot * normal[1]
                    dir_z -= dot * normal[2]
                    norm2 = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
                    dir_x /= norm2
                    dir_y /= norm2
                    dir_z /= norm2

                    # 回転行列生成（Z軸=dir, Y軸=up, X軸=up×Z）
                    up = [0.0, 1.0, 0.0]
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
                        [rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], 0.0],
                        [rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], 0.0],
                        [rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], 0.0],
                        [0.0, 0.0, 0.0, 1.0]
                    ])

                    pose = Pose()
                    pose.position.x = x
                    pose.position.y = y
                    pose.position.z = z
                    pose.orientation.x = quat[0]
                    pose.orientation.y = quat[1]
                    pose.orientation.z = quat[2]
                    pose.orientation.w = quat[3]
                    pose_array.poses.append(pose)

        self.pose_pub.publish(pose_array)
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = MoveOnSlidingSpheres()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
