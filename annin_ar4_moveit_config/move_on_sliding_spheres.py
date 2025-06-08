import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, OrientationConstraint
from builtin_interfaces.msg import Duration
from rclpy.action import ActionClient
import tf_transformations
import math


class MoveOnSlidingSphere(Node):
    def __init__(self):
        super().__init__('move_on_sliding_sphere')

        self.group_name = 'ar_manipulator'
        self._action_client = ActionClient(self, MoveGroup, 'move_action')
        self._action_client.wait_for_server()

        self.center_base = [0.0, -0.45, 0.20]
        self.center_x = 0.0
        self.radius = 0.04
        self.slide_step = 0.002

        self.arc_points = self.generate_intersection_points()
        self.current_index = 0
        self.trajectory = []  # 軌跡保存

        self.center_marker_pub = self.create_publisher(Marker, 'visualization_marker', 10)
        self.trajectory_marker_pub = self.create_publisher(Marker, 'visualization_marker_traj', 10)

        self.publish_center_marker()
        self.publish_trajectory_marker()

        self.send_next_goal()

    def generate_intersection_points(self):
        center = [self.center_base[0] + self.center_x,
                  self.center_base[1],
                  self.center_base[2]]

        points = []
        num_points = 40
        angle_start = math.pi / 2 - math.pi / 4
        angle_end = math.pi / 2 + math.pi / 4

        for i in range(num_points):
            angle = angle_start + (angle_end - angle_start) * i / (num_points - 1)
            x = center[0] + self.radius * math.cos(angle)
            y = center[1]
            z = center[2] + self.radius * math.sin(angle)

            dx, dy, dz = center[0] - x, center[1] - y, center[2] - z
            direction_norm = math.sqrt(dx**2 + dy**2 + dz**2)
            dx, dy, dz = dx / direction_norm, dy / direction_norm, dz / direction_norm

            quat = self.vector_to_quaternion(dx, dy, dz)

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

    def vector_to_quaternion(self, dx, dy, dz):
        z_axis = [dx, dy, dz]
        up = [0.0, 0.0, 1.0]
        right = [up[1]*dz - up[2]*dy, up[2]*dx - up[0]*dz, up[0]*dy - up[1]*dx]
        norm = math.sqrt(sum(i**2 for i in right)) or 1.0
        right = [r / norm for r in right]

        up_corrected = [dz*right[1] - dy*right[2],
                        dx*right[2] - dz*right[0],
                        dy*right[0] - dx*right[1]]

        rot_matrix = [
            [right[0], up_corrected[0], dx],
            [right[1], up_corrected[1], dy],
            [right[2], up_corrected[2], dz]
        ]
        quat = tf_transformations.quaternion_from_matrix([[*row, 0.0] for row in rot_matrix] + [[0.0, 0.0, 0.0, 1.0]])
        return quat

    def send_next_goal(self):
        if self.current_index >= len(self.arc_points):
            self.get_logger().info('✅ All points reached.')
            rclpy.shutdown()
            return

        self.center_x = self.slide_step * self.current_index
        self.arc_points = self.generate_intersection_points()
        pose = self.arc_points[self.current_index % len(self.arc_points)]

        self.publish_center_marker()

        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self.group_name
        goal_msg.request.allowed_planning_time = 5.0
        goal_msg.request.num_planning_attempts = 1
        goal_msg.request.max_velocity_scaling_factor = 0.2
        goal_msg.request.max_acceleration_scaling_factor = 0.2
        goal_msg.request.goal_constraints.append(self.create_constraints(pose))

        self.get_logger().info(f'▶️ Sending goal {self.current_index + 1}/{len(self.arc_points)}')
        self._send_goal_future = self._action_client.send_goal_async(goal_msg)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

        # 軌跡に追加 & 可視化
        self.trajectory.append(pose.pose.position)
        self.publish_trajectory_marker()

    def create_constraints(self, pose):
        constraints = Constraints()
        ocm = OrientationConstraint()
        ocm.link_name = 'ee_link'
        ocm.header.frame_id = pose.header.frame_id
        ocm.orientation = pose.pose.orientation
        ocm.absolute_x_axis_tolerance = 0.1
        ocm.absolute_y_axis_tolerance = 0.1
        ocm.absolute_z_axis_tolerance = 0.1
        ocm.weight = 1.0
        constraints.orientation_constraints.append(ocm)
        return constraints

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('❌ Goal rejected')
            self.current_index += 1
            self.send_next_goal()
            return

        self.get_logger().info('✅ Goal accepted, waiting for result...')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        if result.error_code.val == result.error_code.SUCCESS:
            self.get_logger().info('🎉 Motion succeeded.')
        else:
            self.get_logger().warn(f'⚠️ Motion failed with code: {result.error_code.val}')
        self.current_index += 1
        self.send_next_goal()

    def publish_center_marker(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'sphere_center'
        marker.id = 1000
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.center_base[0] + self.center_x
        marker.pose.position.y = self.center_base[1]
        marker.pose.position.z = self.center_base[2]
        marker.scale.x = self.radius * 2
        marker.scale.y = self.radius * 2
        marker.scale.z = self.radius * 2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.6
        self.center_marker_pub.publish(marker)

    def publish_trajectory_marker(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'ee_trajectory'
        marker.id = 2000
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.005
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.points = [Point(x=p.x, y=p.y, z=p.z) for p in self.trajectory]
        self.trajectory_marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = MoveOnSlidingSphere()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
