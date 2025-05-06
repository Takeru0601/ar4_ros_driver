import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
import tf_transformations
import math
import time

class ReachableArrowVisualizer(Node):
    def __init__(self):
        super().__init__('reachable_arrow_visualizer')

        # Parameters
        self.center = [0.0, -0.33, 0.35]
        self.radius_threshold = 0.4
        self.num_points = 200

        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        self.ik_client = self.create_client(GetPositionIK, '/compute_ik')
        while not self.ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for IK service...')

        self.valid_marker_id = 0
        self.generate_and_check_points()
        self.publish_sphere_marker()

        self.get_logger().info('✅ Done')
        rclpy.shutdown()

    def generate_and_check_points(self):
        phi_steps = int(math.sqrt(self.num_points))
        theta_steps = self.num_points // phi_steps

        for i in range(phi_steps):
            phi = math.pi * i / phi_steps
            for j in range(theta_steps):
                theta = 2 * math.pi * j / theta_steps

                x = self.center[0] + self.radius_threshold * math.sin(phi) * math.cos(theta)
                y = self.center[1] + self.radius_threshold * math.sin(phi) * math.sin(theta)
                z = self.center[2] + self.radius_threshold * math.cos(phi)

                direction = [
                    self.center[0] - x,
                    self.center[1] - y,
                    self.center[2] - z,
                ]
                norm = math.sqrt(sum(d**2 for d in direction))
                z_axis = [d / norm for d in direction]

                pose = PoseStamped()
                pose.header.frame_id = 'base_link'
                pose.pose.position.x = x
                pose.pose.position.y = y
                pose.pose.position.z = z

                quat = self.get_quaternion_facing_z(z_axis)
                pose.pose.orientation.x = quat[0]
                pose.pose.orientation.y = quat[1]
                pose.pose.orientation.z = quat[2]
                pose.pose.orientation.w = quat[3]

                if self.check_ik(pose):
                    self.publish_arrow_marker(pose, z_axis, reachable=True)
                else:
                    self.publish_arrow_marker(pose, z_axis, reachable=False)

                time.sleep(0.01)

    def check_ik(self, pose):
        request = GetPositionIK.Request()
        request.ik_request.group_name = 'ar_manipulator'
        request.ik_request.pose_stamped = pose
        request.ik_request.ik_link_name = 'ee_link'
        request.ik_request.timeout.sec = 1

        future = self.ik_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            return future.result().error_code.val == 1
        return False

    def get_quaternion_facing_z(self, z_axis):
        up = [0, 1, 0]
        x_axis = [
            up[1]*z_axis[2] - up[2]*z_axis[1],
            up[2]*z_axis[0] - up[0]*z_axis[2],
            up[0]*z_axis[1] - up[1]*z_axis[0],
        ]
        x_norm = math.sqrt(sum(v**2 for v in x_axis))
        x_axis = [v / x_norm for v in x_axis]
        new_y = [
            z_axis[1]*x_axis[2] - z_axis[2]*x_axis[1],
            z_axis[2]*x_axis[0] - z_axis[0]*x_axis[2],
            z_axis[0]*x_axis[1] - z_axis[1]*x_axis[0],
        ]
        rot_matrix = [
            [x_axis[0], new_y[0], z_axis[0], 0],
            [x_axis[1], new_y[1], z_axis[1], 0],
            [x_axis[2], new_y[2], z_axis[2], 0],
            [0, 0, 0, 1],
        ]
        return tf_transformations.quaternion_from_matrix(rot_matrix)

    def publish_arrow_marker(self, pose, z_axis, reachable=True):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'reachable_arrows'
        marker.id = self.valid_marker_id
        self.valid_marker_id += 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.005
        marker.scale.y = 0.01
        marker.scale.z = 0.01

        if reachable:
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
            marker.color.a = 1.0
        else:
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

        start = Point()
        start.x = pose.pose.position.x
        start.y = pose.pose.position.y
        start.z = pose.pose.position.z

        end = Point()
        end.x = start.x + z_axis[0] * 0.05
        end.y = start.y + z_axis[1] * 0.05
        end.z = start.z + z_axis[2] * 0.05

        marker.points.append(start)
        marker.points.append(end)

        self.marker_pub.publish(marker)

    def publish_sphere_marker(self):
        marker = Marker()
        marker.header.frame_id = 'base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'arc_volume'
        marker.id = 9999
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = self.center[0]
        marker.pose.position.y = self.center[1]
        marker.pose.position.z = self.center[2]
        marker.scale.x = self.radius_threshold * 2
        marker.scale.y = self.radius_threshold * 2
        marker.scale.z = self.radius_threshold * 2
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.2
        self.marker_pub.publish(marker)

def main(args=None):
    rclpy.init(args=args)
    node = ReachableArrowVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
