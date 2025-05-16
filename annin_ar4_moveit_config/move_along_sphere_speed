#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import tf_transformations
from geometry_msgs.msg import PoseStamped, Point
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from moveit_msgs.msg import PositionIKRequest

class MoveOnSphereTraj(Node):
    def __init__(self):
        super().__init__('move_on_sphere_traj')
        self.center = [0.0, -0.45, 0.20]
        self.radius = 0.1
        self.y_planes = [-0.45, -0.43, -0.41, -0.39, -0.37, -0.35]
        self.tcp_speed = 0.05  # [m/s]

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.traj_pub = self.create_publisher(JointTrajectory, '/joint_trajectory', 10)
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)

        self.get_logger().info('🔄 Waiting for IK service...')
        self.ik_client.wait_for_service()

        self.pose_list = self.generate_intersection_points()
        self.publish_center_marker()
        self.publish_arc_line_marker()

        self.send_trajectory()

    def generate_intersection_points(self):
        points = []
        steps = 18
        for plane_idx, y in enumerate(self.y_planes):
            dy = y - self.center[1]
            if abs(dy) > self.radius:
                continue
            circle_radius = math.sqrt(self.radius**2 - dy**2)

            for i in range(steps + 1):
                theta = math.radians(180 * i / steps) if plane_idx % 2 == 0 else math.radians(180 * (steps - i) / steps)
                x = self.center[0] + circle_radius * math.cos(theta)
                z = self.center[2] + circle_radius * math.sin(theta)

                # 姿勢計算（EEのZ軸が球の中心を向く）
                dir_x = self.center[0] - x
                dir_y = self.center[1] - y
                dir_z = self.center[2] - z
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
        return points

    def send_trajectory(self):
        traj = JointTrajectory()
        traj.joint_names = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        time_from_start = 0.0
        prev_pose = None

        for i, pose in enumerate(self.pose_list):
            req = PositionIKRequest()
            req.group_name = 'ar_manipulator'
            req.robot_state.is_diff = True
            req.ik_link_name = 'ee_link'
            req.pose_stamped = pose

            srv = GetPositionIK.Request()
            srv.ik_request = req

            future = self.ik_client.call_async(srv)
            rclpy.spin_until_future_complete(self, future)

            if not future.result() or future.result().error_code.val != 1:
                self.get_logger().warn(f'❌ IK failed at point {i}')
                continue

            joint_state = future.result().solution.joint_state
            point = JointTrajectoryPoint()
            point.positions = joint_state.position[:6]

            if prev_pose:
                dx = pose.pose.position.x - prev_pose.pose.position.x
                dy = pose.pose.position.y - prev_pose.pose.position.y
                dz = pose.pose.position.z - prev_pose.pose.position.z
                distance = math.sqrt(dx**2 + dy**2 + dz**2)
                dt = distance / self.tcp_speed
                time_from_start += dt
            else:
                time_from_start = 0.0

            point.time_from_start.sec = int(time_from_start)
            point.time_from_start.nanosec = int((time_from_start % 1.0) * 1e9)
            traj.points.append(point)
            prev_pose = pose

        self.traj_pub.publish(traj)
        self.get_logger().info('🚀 Trajectory sent!')

    def publish_center_marker(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'center'
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.center[0]
        marker.pose.position.y = self.center[1]
        marker.pose.position.z = self.center[2]
        marker.scale.x = marker.scale.y = marker.scale.z = 0.02
        marker.color.r = 1.0
        marker.color.a = 1.0
        self.marker_pub.publish(marker)

    def publish_arc_line_marker(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'arc_line'
        marker.id = 2
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.005
        marker.color.g = 1.0
        marker.color.a = 1.0
        for pose in self.pose_list:
            pt = Point()
            pt.x = pose.pose.position.x
            pt.y = pose.pose.position.y
            pt.z = pose.pose.position.z
            marker.points.append(pt)
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = MoveOnSphereTraj()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

