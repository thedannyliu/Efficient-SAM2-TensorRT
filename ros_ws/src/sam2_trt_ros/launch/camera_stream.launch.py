from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("bundle_dir"),
            DeclareLaunchArgument("precision", default_value="fp32"),
            DeclareLaunchArgument("image_topic", default_value="/camera/camera/color/image_raw"),
            DeclareLaunchArgument("max_objects", default_value="8"),
            DeclareLaunchArgument("track_concurrency", default_value="8"),
            DeclareLaunchArgument("trace_path", default_value=""),
            DeclareLaunchArgument("preview_width", default_value="640"),
            DeclareLaunchArgument("preview_height", default_value="360"),
            ComposableNodeContainer(
                name="sam2_trt_container",
                namespace="",
                package="rclcpp_components",
                executable="component_container_mt",
                output="screen",
                composable_node_descriptions=[
                    ComposableNode(
                        package="sam2_trt_ros",
                        plugin="Sam2TrtNode",
                        name="sam2_trt",
                        parameters=[
                            {
                                "bundle_dir": LaunchConfiguration("bundle_dir"),
                                "precision": LaunchConfiguration("precision"),
                                "image_topic": LaunchConfiguration("image_topic"),
                                "queue_policy": "latest",
                                "max_objects": ParameterValue(
                                    LaunchConfiguration("max_objects"),
                                    value_type=int,
                                ),
                                "track_concurrency": ParameterValue(
                                    LaunchConfiguration("track_concurrency"),
                                    value_type=int,
                                ),
                                "trace_path": LaunchConfiguration("trace_path"),
                                "preview_width": ParameterValue(
                                    LaunchConfiguration("preview_width"),
                                    value_type=int,
                                ),
                                "preview_height": ParameterValue(
                                    LaunchConfiguration("preview_height"),
                                    value_type=int,
                                ),
                                "enable_overlay": False,
                            }
                        ],
                        extra_arguments=[{"use_intra_process_comms": True}],
                    )
                ],
            ),
        ]
    )
