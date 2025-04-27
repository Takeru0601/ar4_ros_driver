import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import Constraints, OrientationConstraint
from trajectory_msgs.msg import JointTrajectory
import tf_transformations
from rclpy.action import ActionClient
from builtin_interfaces.msg import Duration

class MoveAlongArc(Node):
    def __init__(self):
        super().__init__('move_along_arc')
        self.client = ActionClient(self, MoveGroup, '/move_action')

        self.center = np.array([0.0, -0.35, 0.35])  # 中心点
        self.radius = 0.30                          # 半径
        self.height = 0.4                           # 円弧の高さ（Z座標）
        self.start_angle = np.radians(0)             # 開始角度
        self.end_angle = np.radians(90)              # 終了角度
        self.steps = 10                              # 何分割するか（ステップ数）

        self.group_name = 'ar_manipulator'

    def compute_orientation_towards_center(self, position, center):
        dir_vec = np.array(center) - np.array(position)
        dir_vec /= np.linalg.norm(dir_vec)

        up_vec = np.array([0.0, 1.0, 0.0])

        x_axis = np.cross(up_vec, dir_vec)
        if np.linalg.norm(x_axis) < 1e-6:
            up_vec = np.array([1.0, 0.0, 0.0])
            x_axis = np.cross(up_vec, dir_vec)
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(dir_vec, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        rot_matrix = np.eye(4)
        rot_matrix[0, :3] = x_axis
        rot_matrix[1, :3] = y_axis
        rot_matrix[2, :3] = dir_vec

        quat = tf_transformations.quaternion_from_matrix(rot_matrix)
        return quat

    def create_pose(self, x, y, z, quat):
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

    def send_goal(self, pose):
        goal_msg = MoveGroup.Goal()
        goal_msg.request.group_name = self.group_name
        goal_msg.request.allowed_planning_time = 5.0

        goal_msg.request.goal_constraints.append(Constraints())
        goal_msg.request.goal_constraints[0].position_constraints = []
        goal_msg.request.goal_constraints[0].orientation_constraints = []

        goal_msg.request.pose_goal = pose.pose

        self.client.wait_for_server()

        future = self.client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()

        if result.error_code.val == 1:
            self.get_logger().info('✅ Move succeeded!')
        else:
            self.get_logger().error(f'❌ Move failed with code: {result.error_code.val}')

    def move_along_arc(self):
        angles = np.linspace(self.start_angle, self.end_angle, self.steps)

        for theta in angles:
            x = self.center[0] + self.radius * np.cos(theta)
            y = self.center[1] + self.radius * np.sin(theta)
            z = self.height

            position = [x, y, z]
            quat = self.compute_orientation_towards_center(position, self.center)
            pose = self.create_pose(x, y, z, quat)

            self.send_goal(pose)

def main(args=None):
    rclpy.init(args=args)
    node = MoveAlongArc()
    node.move_along_arc()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
