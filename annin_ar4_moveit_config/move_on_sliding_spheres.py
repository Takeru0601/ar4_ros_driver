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

        # 球の初期中心 (xは0から動かす)
        self.base_center = [0.0, -0.45, 0.20]
        self.radius = 0.4  # 直径80cm
        self.y_planes = [-0.45, -0.43, -0.41, -0.39, -0.37, -0.35]
        self.steps = 18

        self.arc_points = self.generate_sliding_arc_points()
        self.current_index = 0

        self.get_logger().info('Waiting for MoveGroup action server...')
        self._action_client.wait_for_server()

        self.publish_arc_line_marker()
        self.send_next_goal()

    def generate_sliding_arc_points(self):
        points = []
        for plane_idx, y in enumerate(self.y_planes):
            dy = y - self.base_center[1]
            if abs(dy) > self.radius:
                continue
            r_slice = math.sqrt(self.radius**2 - dy**2)

            for i in range(self.steps + 1):
                if plane_idx % 2 == 0:
                    theta = math.radians(180 * i / self.steps)
                else:
                    theta = math.radians(180 * (self.steps - i) / self.steps)

                x_offset = 0.15 * (plane_idx + i / self.steps)  # 球の中心をx方向にスライド
                center = [self.base_center[0] + x_offset, self.base_center[1], self.base_center[2]]

                x = center[0] + r_slice * math.cos(theta)
                z = center[2] + r_slice * math.sin(theta)

                dir_x = center[0] - x
                dir_y = center[1] - y
                dir_z = center[2] - z
                norm = math.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
                dir_x /= norm
                dir_y /= norm
                dir_z /= norm

                up = [0.0, 1.0, 0.0]
                x_axis = [
                    up[1]*dir_z - up[2]*dir_y,
                    up[2]*dir_x - up[0]*dir_z,
                    up[0]*dir_y - up[1]*dir_x
                ]
                x_norm = math.sqrt(sum(v**2 for v in x_axis))
                x_axis = [v / x_norm for v in x_axis]
                y_axis = [
                    dir_y*x_axis[2] - dir_z*x_axis[1],
                    dir_z*x_axis[0] - dir_x*x_axis[2],
                    dir_x*x_axis[1] - dir_y*x_axis[0]
                ]

                rot_matrix = [
                    [x_axis[0], y_axis[0], dir_x],
                    [x_axis[1], y_axis[1], dir_y],
                    [x_axis[2], y_axis[2], dir_z]
                ]
                quat = tf_transformations.quaternion_from_matrix([
                    [*rot_matrix[0], 0.0],
                    [*rot_matrix[1], 0.0],
                    [*rot_matrix[2], 0.0],
                    [0.0, 0.0, 0.0, 1.0]
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

    def send_next_goal(self):
        if self.current_index >= len(self.arc_points):
            self.get_logger().info('✅ All points executed!')
            rclpy.shutdown()
            return

        pose = self.arc_points[self.current_index]
        self.get_logger().info(f'▶️ Point {self.current_index + 1}/{len(self.arc_points)}')
        self.publish_ee_z_axis(pose, self.current_index)

        goal_msg = MoveGroup.Goal()
        req = MotionPlanRequest()
        req.group_name = 'ar_manipulator'
        req.max_velocity_scaling_factor = 0.3
        req.max_acceleration_scaling_factor = 0.3

        pc = PositionConstraint()
        pc.header.frame_id = pose.header.frame_id
        pc.link_name = 'ee_link'
        pc.target_point_offset.x = 0.0
        pc.target_point_offset.y = 0.0
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
        oc.absolute_x_axis_tolerance = 0.01
        oc.absolute_y_axis_tolerance = 0.01
        oc.absolute_z_axis_tolerance = 0.01
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
        self.get_logger().info('✅ Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'🎯 Result: {result.error_code}')
        self.current_index += 1
        self.send_next_goal()

    def publish_ee_z_axis(self, pose, id_num):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'ee_z_axis'
        marker.id = id_num
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.scale.y = 0.015
        marker.scale.z = 0.1
        marker.color.b = 1.0
        marker.color.a = 1.0

        start = Point()
        start.x = pose.pose.position.x
        start.y = pose.pose.position.y
        start.z = pose.pose.position.z

        quat = (
            pose.pose.orientation.x,
            pose.pose.orientation.y,
            pose.pose.orientation.z,
            pose.pose.orientation.w
        )
        rot_matrix = tf_transformations.quaternion_matrix(quat)
        z_axis = [rot_matrix[0][2], rot_matrix[1][2], rot_matrix[2][2]]

        end = Point()
        end.x = start.x + 0.05 * z_axis[0]
        end.y = start.y + 0.05 * z_axis[1]
        end.z = start.z + 0.05 * z_axis[2]

        marker.points.append(start)
        marker.points.append(end)
        self.marker_pub.publish(marker)

    def publish_arc_line_marker(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'trajectory_line'
        marker.id = 999
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.005
        marker.color.g = 1.0
        marker.color.a = 1.0

        for pose in self.arc_points:
            pt = Point()
            pt.x = pose.pose.position.x
            pt.y = pose.pose.position.y
            pt.z = pose.pose.position.z
            marker.points.append(pt)

        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = MoveOnSlidingSphere()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
