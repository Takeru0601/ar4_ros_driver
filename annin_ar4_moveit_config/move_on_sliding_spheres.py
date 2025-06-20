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
        self.arc_points_and_centers = self.generate_intersection_points_with_dynamic_slide()  # NOTE: Feasibility check will stamp header inside
        self.get_logger().info(f'✅ {len(self.arc_points_and_centers)} feasible points found.')

        self.current_index = 0
        self.get_logger().info('⏳ Waiting for MoveGroup action server...')
        self._action_client.wait_for_server()
        self.get_logger().info('✅ MoveGroup action server connected.')
        self.send_next_goal()

    def compute_pose_pointing_to_center(self, x, y, z, cx, cy, cz):
        # NOTE: PoseStamped will be stamped in quick_feasibility_check
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
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'base_link'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = quat[0]
        pose.pose.orientation.y = quat[1]
        pose.pose.orientation.z = quat[2]
        pose.pose.orientation.w = quat[3]
        return pose

    def send_next_goal(self):
        if self.current_index >= len(self.arc_points_and_centers):
            self.get_logger().info('🎉 All feasible points executed.')
            self.publish_trajectories()
            rclpy.shutdown()
            return

        pose, center = self.arc_points_and_centers[self.current_index]
        self.ee_traj.append(pose.pose.position)
        self.sphere_traj.append(center)
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
        marker.color.r = 0.2
        marker.color.g = 0.5
        marker.color.b = 1.0
        marker.color.a = 0.4
        self.marker_pub.publish(marker)

    def publish_trajectories(self):
        if not self.ee_traj or not self.sphere_traj:
            self.get_logger().warn("❗ Trajectories are empty.")
            return
        marker_array = MarkerArray()

        ee_line = Marker()
        ee_line.header.frame_id = 'base_link'
        ee_line.ns = 'ee_trajectory'
        ee_line.id = 0
        ee_line.type = Marker.LINE_STRIP
        ee_line.action = Marker.ADD
        ee_line.scale.x = 0.005
        ee_line.color.g = 1.0
        ee_line.color.a = 1.0
        ee_line.points = [Point(x=p.x, y=p.y, z=p.z) for p in self.ee_traj]

        sphere_line = Marker()
        sphere_line.header.frame_id = 'base_link'
        sphere_line.ns = 'sphere_trajectory'
        sphere_line.id = 1
        sphere_line.type = Marker.LINE_STRIP
        sphere_line.action = Marker.ADD
        sphere_line.scale.x = 0.003
        sphere_line.color.b = 1.0
        sphere_line.color.a = 1.0
        sphere_line.points = [Point(x=p[0], y=p[1], z=p[2]) for p in self.sphere_traj]

        marker_array.markers.append(ee_line)
        marker_array.markers.append(sphere_line)
        self.marker_array_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = MoveOnSlidingSphere()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
