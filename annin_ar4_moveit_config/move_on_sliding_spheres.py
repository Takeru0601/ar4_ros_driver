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

        # 初期設定
        self.center_base = [0.0, -0.45, 0.0]  # 初期中心 x=0
        self.radius = 0.2  # 半径 0.2m
        self.y_planes = [-0.45, -0.43, -0.41, -0.39, -0.37, -0.35]  # Y軸に沿った平面

        # 目標点と中心点
        self.ee_traj = []
        self.sphere_traj = []

        # 計算された交差点と中心を取得
        self.get_logger().info('⏳ Generating feasible target points...')
        self.arc_points_and_centers = self.generate_intersection_points_with_dynamic_slide()
        self.get_logger().info(f'✅ {len(self.arc_points_and_centers)} feasible points found.')

        self.current_index = 0

        # MoveGroupアクションサーバーを待機
        self.get_logger().info('⏳ Waiting for MoveGroup action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('✅ MoveGroup action server connected.')

        self.send_next_goal()

    def generate_intersection_points_with_dynamic_slide(self):
        points = []
        steps = 18  # 角度を18等分
        max_slide = 0.2  # 最大スライド量
        slide_resolution = 0.01  # スライド解像度
        slide_attempts = int(max_slide / slide_resolution)

        # 各Y平面に対して交差点を計算
        for plane_idx, y in enumerate(self.y_planes):
            dy = y - self.center_base[1]
            if abs(dy) > self.radius:
                continue
            circle_radius = math.sqrt(self.radius**2 - dy**2)

            # 各角度で点を計算
            for i in range(steps + 1):
                theta = math.radians(180 * i / steps if plane_idx % 2 == 0 else 180 * (steps - i) / steps)

                for j in range(-slide_attempts, slide_attempts + 1):
                    x_slide = j * slide_resolution
                    center_x = self.center_base[0] + x_slide
                    center_y = self.center_base[1]
                    center_z = self.center_base[2]

                    # 角度に基づいて位置を計算
                    x = center_x + circle_radius * math.cos(theta)
                    z = center_z + circle_radius * math.sin(theta)

                    # 中心点を向く姿勢を計算
                    pose = self.compute_pose_pointing_to_center(x, y, z, center_x, center_y, center_z)

                    if self.quick_feasibility_check(pose):
                        points.append((pose, [center_x, center_y, center_z]))
                        break
                else:
                    self.get_logger().warn(f'❌ IK failed at y={y:.3f}, θ={math.degrees(theta):.1f}')
        return points

    def compute_pose_pointing_to_center(self, x, y, z, cx, cy, cz):
        # 中心を向く方向の計算
        dir_x, dir_y, dir_z = cx - x, cy - y, cz - z
        norm = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
        dir_x /= norm
        dir_y /= norm
        dir_z /= norm

        # X軸の方向を計算
        up = [0, 1, 0]
        x_axis = [
            up[1]*dir_z - up[2]*dir_y,
            up[2]*dir_x - up[0]*dir_z,
            up[0]*dir_y - up[1]*dir_x,
        ]
        x_norm = math.sqrt(sum(v**2 for v in x_axis))
        x_axis = [v / x_norm for v in x_axis]

        # 新しいY軸を計算
        new_y = [
            dir_y*x_axis[2] - dir_z*x_axis[1],
            dir_z*x_axis[0] - dir_x*x_axis[2],
            dir_x*x_axis[1] - dir_y*x_axis[0],
        ]

        # 回転行列を作成
        rot_matrix = [
            [x_axis[0], new_y[0], dir_x],
            [x_axis[1], new_y[1], dir_y],
            [x_axis[2], new_y[2], dir_z],
        ]
        quat = tf_transformations.quaternion_from_matrix([  # 回転行列をクォータニオンに変換
            [rot_matrix[0][0], rot_matrix[0][1], rot_matrix[0][2], 0],
            [rot_matrix[1][0], rot_matrix[1][1], rot_matrix[1][2], 0],
            [rot_matrix[2][0], rot_matrix[2][1], rot_matrix[2][2], 0],
            [0, 0, 0, 1]
        ])

        # 姿勢を生成
        pose = PoseStamped()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        return pose

    def quick_feasibility_check(self, pose):
        # クイックな可行性チェック（ここでは常にTrueとしている）
        return True

    def send_next_goal(self):
        if self.current_index >= len(self.arc_points_and_centers):
            self.get_logger().info('🎉 All feasible points executed.')
            rclpy.shutdown()
            return

        pose, center = self.arc_points_and_centers[self.current_index]
        self.publish_center_marker(center, self.current_index)

        # MoveGroupへ目標メッセージを送信
        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = 'ar_manipulator'  # グループ名設定
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3
        req.start_state.is_diff = True

        pc = PositionConstraint()
        pc.header.frame_id = pose.header.frame_id
        pc.link_name = 'ee_link'
        pc.target_point_offset.z = 0.0
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.01, 0.01, 0.01]
        pc.constraint_region.primitives.append(box)
        pc.constraint_region.primitive_poses.append(pose.pose)

        oc = OrientationConstraint()
        oc.header.frame_id = pose.header.frame_id
        oc.link_name = 'ee_link'
        oc.orientation = pose.pose.orientation
        oc.absolute_x_axis_tolerance = 0.3
        oc.absolute_y_axis_tolerance = 0.3
        oc.absolute_z_axis_tolerance = 0.3
        oc.weight = 1.0

        constraints = Constraints()
        constraints.position_constraints.append(pc)
        constraints.orientation_constraints.append(oc)
        req.goal_constraints.append(constraints)
        goal_msg.request = req

        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ Goal rejected')
            rclpy.shutdown()
            return
        self.get_logger().info(f'▶️ Executing point {self.current_index + 1}/{len(self.arc_points_and_centers)}')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'🎯 Result code: {result.error_code.val}')
        self.current_index += 1
        self.send_next_goal()

    def publish_center_marker(self, center, marker_id):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'sphere_center'
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]
        marker.scale.x = self.radius * 2
        marker.scale.y = self.radius * 2
        marker.scale.z = self.radius * 2
        marker.color.r = 1.0
        marker.color.g = 0.3
        marker.color.b = 0.3
        marker.color.a = 0.5
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = MoveOnSlidingSphere()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
