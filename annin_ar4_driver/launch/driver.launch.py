from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare

from launch.conditions import IfCondition
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.substitutions import LaunchConfiguration
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch.actions import TimerAction
from launch.event_handlers import OnProcessStart


def generate_launch_description():
    serial_port = LaunchConfiguration("serial_port")
    calibrate = LaunchConfiguration("calibrate")
    include_gripper = LaunchConfiguration("include_gripper")
    arduino_serial_port = LaunchConfiguration("arduino_serial_port")
    ar_model_config = LaunchConfiguration("ar_model")
    tf_prefix = LaunchConfiguration("tf_prefix")
    controller_update_rate = LaunchConfiguration("controller_update_rate")

    # ===============
    # robot_description
    # ===============
    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution(
            [FindPackageShare("annin_ar4_driver"), "urdf", "ar.urdf.xacro"]),
        " ",
        "ar_model:=", ar_model_config, " ",
        "serial_port:=", serial_port, " ",
        "calibrate:=", calibrate, " ",
        "tf_prefix:=", tf_prefix, " ",
        "include_gripper:=", include_gripper, " ",
        "arduino_serial_port:=", arduino_serial_port,
    ])
    robot_description = {"robot_description": robot_description_content}

    # ===============
    # ros2_control params
    # ===============
    joint_controllers_cfg = PathJoinSubstitution([
        FindPackageShare("annin_ar4_driver"), "config", "controllers.yaml"
    ])

    controller_manager_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            # 安定化のため update_rate を起動引数から直接指定（YAMLが無くても動作）
            {"update_rate": controller_update_rate},
            ParameterFile(joint_controllers_cfg, allow_substs=True),
            {"tf_prefix": tf_prefix},
        ],
        remappings=[('~/robot_description', 'robot_description')],
        output="screen",
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    # ===============
    # Spawners (直列化 & 遅延)
    # ===============
    spawner_jsb = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "-c", "/controller_manager",
            "--controller-manager-timeout", "120",
        ],
        output="screen",
    )

    spawner_jtc = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_trajectory_controller",
            "-c", "/controller_manager",
            "--controller-manager-timeout", "120",
        ],
        output="screen",
    )

    spawner_gripper = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "gripper_controller",
            "-c", "/controller_manager",
            "--controller-manager-timeout", "60",
        ],
        condition=IfCondition(include_gripper),
        output="screen",
    )

    # controller_manager の起動をトリガにして順番にスポーン
    sequenced_spawners = RegisterEventHandler(
        OnProcessStart(
            target_action=controller_manager_node,
            on_start=[
                # 1) JSB を先に
                TimerAction(period=3.0, actions=[spawner_jsb]),
                # 2) 少し待って JTC
                TimerAction(period=6.0, actions=[spawner_jtc]),
                # 3) さらに必要ならグリッパ
                TimerAction(period=9.0, actions=[spawner_gripper]),
            ],
        )
    )

    # ===============
    # LaunchDescription
    # ===============
    ld = LaunchDescription()
    ld.add_action(DeclareLaunchArgument(
        "serial_port", default_value="/dev/ttyACM0",
        description="Serial port to connect to the robot",
    ))
    ld.add_action(DeclareLaunchArgument(
        "calibrate", default_value="False",  # 実機運用を優先してデフォルト False
        description="Calibrate the robot on startup",
        choices=["True", "False"],
    ))
    ld.add_action(DeclareLaunchArgument(
        "tf_prefix", default_value="",
        description="Prefix for AR4 tf_tree",
    ))
    ld.add_action(DeclareLaunchArgument(
        "include_gripper", default_value="False",
        description="Run the servo gripper",
        choices=["True", "False"],
    ))
    ld.add_action(DeclareLaunchArgument(
        "arduino_serial_port", default_value="/dev/ttyUSB0",
        description="Serial port of the Arduino nano for the servo gripper",
    ))
    ld.add_action(DeclareLaunchArgument(
        "ar_model", default_value="mk3",
        choices=["mk1", "mk2", "mk3"], description="Model of AR4",
    ))
    ld.add_action(DeclareLaunchArgument(
        "controller_update_rate", default_value="100",  # VM/USBで安定する値
        description="controller_manager update rate [Hz]",
    ))

    # 起動順
    ld.add_action(controller_manager_node)
    ld.add_action(robot_state_publisher_node)
    ld.add_action(sequenced_spawners)

    return ld
