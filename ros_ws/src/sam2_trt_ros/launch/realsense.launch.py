from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("bundle_dir"),
            DeclareLaunchArgument("precision", default_value="fp32"),
            DeclareLaunchArgument("image_topic", default_value="/camera/camera/color/image_raw"),
            Node(
                package="sam2_trt_ros",
                executable="sam2_trt_node",
                output="screen",
                parameters=[
                    {
                        "bundle_dir": LaunchConfiguration("bundle_dir"),
                        "precision": LaunchConfiguration("precision"),
                        "image_topic": LaunchConfiguration("image_topic"),
                        "queue_policy": "latest",
                        "max_objects": 8,
                        "enable_overlay": False,
                    }
                ],
            ),
        ]
    )
