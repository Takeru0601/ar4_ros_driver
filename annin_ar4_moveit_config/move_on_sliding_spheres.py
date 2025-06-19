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
from visualization_msgs.msg import Marker, MarkerArray


class MoveOnSlidingSphere(Node):
    def __init__(self):
        super().__init__('move_on_sliding_sphere')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        self.marker_array_pub = self.create_publisher(MarkerArray, '/visualization_marker_array', 10)

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
                        self.ee_traj.append((x, y, z))
                        self.sphere_traj.append((center_x, center_y, center_z))
                        self.center_base_x = center_x
                        self.get_logger().info(
                            f'✅ IK success: y={y:.3f}, θ={theta_deg:5.1f}°, x_slide={x_slide:+.3f} m'
                        )
                        break
                else:
                    self.get_logger().warn(f'❌ IK failed at y={y:.3f}, θ={theta_deg:.1f}')
        return points

    def send_next_goal(self):
        if self.current_index >= len(self.arc_points_and_centers):
            self.get_logger().info('🎉 All feasible points executed.')
            self.publish_trajectories()
            rclpy.shutdown()
            return

        pose, center = self.arc_points_and_centers[self.current_index]
        self.publish_sphere_marker(center, self.current_index)

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = 'ar_manipulator'
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

    def publish_sphere_marker(self, center, marker_id):
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
        marker.color.r = 0.3
        marker.color.g = 0.3
        marker.color.b = 1.0
        marker.color.a = 0.4
        self.marker_pub.publish(marker)

    def publish_trajectories(self):
        marker_array = MarkerArray()

        line_marker = Marker()
        line_marker.header.frame_id = 'base_link'
        line_marker.header.stamp = self.get_clock().now().to_msg()
        line_marker.ns = 'ee_traj'
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.005
        line_marker.color.r = 0.0
        line_marker.color.g = 1.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0
        for pt in self.ee_traj:
            p = Point()
            p.x, p.y, p.z = pt
            line_marker.points.append(p)
        marker_array.markers.append(line_marker)

        sphere_line = Marker()
        sphere_line.header.frame_id = 'base_link'
        sphere_line.header.stamp = self.get_clock().now().to_msg()
        sphere_line.ns = 'sphere_traj'
        sphere_line.id = 1
        sphere_line.type = Marker.LINE_STRIP
        sphere_line.action = Marker.ADD
        sphere_line.scale.x = 0.005
        sphere_line.color.r = 0.0
        sphere_line.color.g = 0.0
        sphere_line.color.b = 1.0
        sphere_line.color.a = 1.0
        for pt in self.sphere_traj:
            p = Point()
            p.x, p.y, p.z = pt
            sphere_line.points.append(p)
        marker_array.markers.append(sphere_line)

        self.marker_array_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = MoveOnSlidingSphere()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
