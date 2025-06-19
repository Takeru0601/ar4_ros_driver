def quick_feasibility_check(self, pose: PoseStamped) -> bool:
    request = GetPositionIK.Request()
    request.ik_request.group_name = 'ar_manipulator'
    request.ik_request.pose_stamped = pose
    request.ik_request.ik_link_name = 'ee_link'
    request.ik_request.timeout.sec = 1  # ← attempts は削除
    request.ik_request.avoid_collisions = True
    request.ik_request.robot_state.is_diff = True

    future = self.ik_client.call_async(request)
    rclpy.spin_until_future_complete(self, future)

    if future.result() is not None:
        result = future.result()
        if result.error_code.val == result.error_code.SUCCESS:
            return True
        else:
            self.get_logger().warn(f'IK failed: error code {result.error_code.val}')
    else:
        self.get_logger().error('Failed to receive IK response.')

    return False
