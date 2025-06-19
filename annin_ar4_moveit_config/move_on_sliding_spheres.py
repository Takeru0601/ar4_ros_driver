#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import math
import tf_transformations

from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import JointState
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import MotionPlanRequest, Constraints, PositionConstraint, OrientationConstraint
from moveit_msgs.srv import GetPositionIK
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker


class MoveOnSlidingSphere(Node):
    def __init__(self):
        super().__init__('move_on_sliding_sphere')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)

        self.center_base_x = 0.0
        self.center_base_y = -0.45
        self.center_base_z = 0.0
        self.radius = 0.2
        self.y_planes = [-0.45, -0.43, -0.41, -0.39, -0.37, -0.35]
        self.ee_traj = []
        self.sphere_traj = []

        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /compute_ik service...')

        self.get_logger().info('⏳ Generating feasible target points...')
        self.arc_points_and_centers = self.generate_intersection_points_with_dynamic_slide()
        self.get_logger().info(f'✅ {len(self.arc_points_and_centers)} feasible points found.')

        self.current_index = 0
        self.get_logger().info('⏳ Waiting for MoveGroup action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('✅ MoveGroup action server connected.')
        self.send_next_goal()

    def quick_feasibility_check(self, pose: PoseStamped) -> bool:
        request = GetPositionIK.Request()
        request.ik_request.group_name = 'ar_manipulator'
        request.ik_request.pose_stamped = pose
        request.ik_request.ik_link_name = 'ee_link'
        request.ik_request.timeout.sec = 1
        request.ik_request.avoid_collisions = True
        request.ik_request.robot_state.is_diff = True

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            result = future.result()
            return result.error_code.val == result.error_code.SUCCESS
        else:
            self.get_logger().warn('No response from /compute_ik')
            return False

    def generate_intersection_points_with_dynamic_slide(self):
        points = []
        steps = 18
        max_slide = 0.2
        slide_resolution = 0.01
        slide_attempts = int(max_slide / slide_resolution)

        for plane_idx, y in enumerate(self.y_planes):
            dy = y - self.center_base_y
            if abs(dy) > self.radius:
                continue
            circle_radius = math.sqrt(self.radius**2 - dy**2)

            for i in range(steps + 1):
                theta_deg = 180 * i / steps if plane_idx % 2 == 0 else 180 * (steps - i) / steps
                theta = math.radians(theta_deg)

                if theta_deg < 90.0:
                    slide_range = range(-slide_attempts, slide_attempts + 1)
                else:
                    slide_range = range(slide_attempts, -slide_attempts - 1, -1)

                for j in slide_range:
                    x_slide = j * slide_resolution
                    center_x = self.center_base_x + x_slide
                    center_y = self.center_base_y
                    center_z = self.center_base_z

                    x = center_x + circle_radius * math.cos(theta)
                    z = center_z + circle_radius * math.sin(theta)

                    pose = self.compute_pose_pointing_to_center(x, y, z, center_x, center_y, center_z)

                    if self.quick_feasibility_check(pose):
                        points.append((pose, [center_x, center_y, center_z]))
                        self.center_base_x = center_x
                        self.get_logger().info(
                            f'✅ IK success: y={y:.3f}, θ={theta_deg:5.1f}°, x_slide={x_slide:+.3f} m'
                        )
                        break
                else:
                    self.get_logger().warn(f'❌ IK failed at y={y:.3f}, θ={theta_deg:.1f}')
        return points

    def compute_pose_pointing_to_center(self, x, y, z, cx, cy, cz):
        dir_x, dir_y, dir_z = cx - x, cy - y, cz - z
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
        return pose

    # ... send_next_goal(), publish_sphere_marker() など他の関数は前回のコードと同様


def main(args=None):
    rclpy.init(args=args)
    node = MoveOnSlidingSphere()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
