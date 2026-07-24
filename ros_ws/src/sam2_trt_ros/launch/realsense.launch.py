from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("bundle_dir"),
            DeclareLaunchArgument("precision", default_value="fp32"),
            DeclareLaunchArgument("image_topic", default_value="/camera/camera/color/image_raw"),
            DeclareLaunchArgument("max_objects", default_value="8"),
            DeclareLaunchArgument("track_concurrency", default_value="8"),
            DeclareLaunchArgument("trace_path", default_value=""),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
                    )
                ),
                launch_arguments={"enable_color": "true", "enable_depth": "false"}.items(),
            ),
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
                        "max_objects": ParameterValue(
                            LaunchConfiguration("max_objects"), value_type=int
                        ),
                        "track_concurrency": ParameterValue(
                            LaunchConfiguration("track_concurrency"), value_type=int
                        ),
                        "trace_path": LaunchConfiguration("trace_path"),
                        "enable_overlay": False,
                    }
                ],
            ),
        ]
    )
