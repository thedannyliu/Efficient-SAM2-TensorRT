from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    image_topic = LaunchConfiguration("image_topic")
    return LaunchDescription(
        [
            DeclareLaunchArgument("bundle_dir"),
            DeclareLaunchArgument("precision", default_value="fp16"),
            DeclareLaunchArgument("image_topic", default_value="/camera/camera/color/image_raw"),
            DeclareLaunchArgument("max_objects", default_value="8"),
            DeclareLaunchArgument("track_concurrency", default_value="8"),
            DeclareLaunchArgument("trace_path", default_value=""),
            DeclareLaunchArgument("display_scale", default_value="1.0"),
            DeclareLaunchArgument("display_max_width", default_value="1280"),
            DeclareLaunchArgument("replace_on_prompt", default_value="true"),
            DeclareLaunchArgument("draw_contours", default_value="false"),
            DeclareLaunchArgument("preview_width", default_value="640"),
            DeclareLaunchArgument("preview_height", default_value="360"),
            DeclareLaunchArgument(
                "color_profile", default_value="1280x720x30"
            ),
            ComposableNodeContainer(
                name="sam2_camera_container",
                namespace="",
                package="rclcpp_components",
                executable="component_container_mt",
                output="screen",
                composable_node_descriptions=[
                    ComposableNode(
                        package="realsense2_camera",
                        plugin="realsense2_camera::RealSenseNodeFactory",
                        namespace="camera",
                        name="camera",
                        parameters=[
                            {
                                "enable_color": True,
                                "enable_depth": False,
                                "enable_infra": False,
                                "enable_infra1": False,
                                "enable_infra2": False,
                                "enable_gyro": False,
                                "enable_accel": False,
                                "enable_sync": False,
                                "rgb_camera.color_profile": LaunchConfiguration(
                                    "color_profile"
                                ),
                                "rgb_camera.color_format": "RGB8",
                            }
                        ],
                        extra_arguments=[{"use_intra_process_comms": True}],
                    ),
                    ComposableNode(
                        package="sam2_trt_ros",
                        plugin="Sam2TrtNode",
                        name="sam2_trt",
                        parameters=[
                            {
                                "bundle_dir": LaunchConfiguration("bundle_dir"),
                                "precision": LaunchConfiguration("precision"),
                                "image_topic": image_topic,
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
                    ),
                ],
            ),
            Node(
                package="sam2_trt_ros",
                executable="sam2_trt_interactive_viewer",
                output="screen",
                parameters=[
                    {
                        "image_topic": image_topic,
                        "display_scale": ParameterValue(
                            LaunchConfiguration("display_scale"), value_type=float
                        ),
                        "display_max_width": ParameterValue(
                            LaunchConfiguration("display_max_width"), value_type=int
                        ),
                        "replace_on_prompt": ParameterValue(
                            LaunchConfiguration("replace_on_prompt"), value_type=bool
                        ),
                        "draw_contours": ParameterValue(
                            LaunchConfiguration("draw_contours"), value_type=bool
                        ),
                        "use_preview": True,
                    }
                ],
            ),
        ]
    )
