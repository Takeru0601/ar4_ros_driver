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
from visualization_msgs.msg import Marker, MarkerArray

class MoveOnSlidingSphere(Node):
    def __init__(self):
        super().__init__('move_on_sliding_sphere')
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        self.traj_marker_pub = self.create_publisher(Marker, '/ee_trajectory_marker', 10)
        self.center_traj_pub = self.create_publisher(Marker, '/sphere_center_traj_marker', 10)
        self.tf_buffer = None
        self.tf_listener = None

        self.center_base = [0.0, -0.45, 0.0]
        self.radius = 0.2
        self.y_planes = [-0.45, -0.43, -0.41, -0.39, -0.37, -0.35]

        self.ee_traj_points = []
        self.center_traj_points = []

        self.get_logger().info('⏳ Generating feasible target points...')
        self.arc_points_and_centers = self.generate_intersection_points_with_dynamic_slide()
        self.get_logger().info(f'✅ {len(self.arc_points_and_centers)} feasible points found.')

        self.current_index = 0

        self.get_logger().info('⏳ Waiting for MoveGroup action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('✅ MoveGroup action server connected.')

        self.send_next_goal()

    def generate_intersection_points_with_dynamic_slide(self):
        points = []
        steps = 18
        max_slide = 0.2
        slide_resolution = 0.01
        slide_attempts = int(max_slide / slide_resolution)

        for plane_idx, y in enumerate(self.y_planes):
            dy = y - self.center_base[1]
            if abs(dy) > self.radius:
                continue
            circle_radius = math.sqrt(self.radius**2 - dy**2)

            for i in range(steps + 1):
                theta = math.radians(180 * i / steps if plane_idx % 2 == 0 else 180 * (steps - i) / steps)

                for j in range(-slide_attempts, slide_attempts + 1):
                    x_slide = j * slide_resolution
                    x = self.center_base[0] + circle_radius * math.cos(theta) + x_slide
                    z = self.center_base[2] + circle_radius * math.sin(theta)

                    center_x = self.center_base[0] + x_slide
                    center_y = self.center_base[1]
                    center_z = self.center_base[2]

                    pose = self.compute_pose_pointing_to_center(x, y, z, center_x, center_y, center_z)

                    if self.quick_feasibility_check(pose):
                        points.append((pose, [center_x, center_y, center_z]))
                        break
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

    def quick_feasibility_check(self, pose):
        return True

    def send_next_goal(self):
        if self.current_index >= len(self.arc_points_and_centers):
            self.publish_trajectory_markers()
            self.get_logger().info('🎉 All feasible points executed.')
            rclpy.shutdown()
            return

        pose, center = self.arc_points_and_centers[self.current_index]
        self.center_traj_points.append(Point(x=center[0], y=center[1], z=center[2]))
        self.ee_traj_points.append(pose.pose.position)
        self.publish_center_marker(center)

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

    def publish_center_marker(self, center):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'sphere_center'
        marker.id = 1000
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

    def publish_trajectory_markers(self):
        traj_marker = Marker()
        traj_marker.header.frame_id = 'base_link'
        traj_marker.header.stamp = self.get_clock().now().to_msg()
        traj_marker.ns = 'ee_trajectory'
        traj_marker.id = 2000
        traj_marker.type = Marker.LINE_STRIP
        traj_marker.action = Marker.ADD
        traj_marker.scale.x = 0.005
        traj_marker.color.r = 0.0
        traj_marker.color.g = 1.0
        traj_marker.color.b = 0.0
        traj_marker.color.a = 1.0
        traj_marker.points = self.ee_traj_points
        self.traj_marker_pub.publish(traj_marker)

        center_marker = Marker()
        center_marker.header.frame_id = 'base_link'
        center_marker.header.stamp = self.get_clock().now().to_msg()
        center_marker.ns = 'sphere_center_traj'
        center_marker.id = 3000
        center_marker.type = Marker.LINE_STRIP
        center_marker.action = Marker.ADD
        center_marker.scale.x = 0.005
        center_marker.color.r = 1.0
        center_marker.color.g = 0.0
        center_marker.color.b = 1.0
        center_marker.color.a = 1.0
        center_marker.points = self.center_traj_points
        self.center_traj_pub.publish(center_marker)

def main(args=None):
    rclpy.init(args=args)
    node = MoveOnSlidingSphere()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
