#pragma once

#include <rclcpp/node.hpp>
#include <rclcpp/node_options.hpp>

#include <memory>

std::shared_ptr<rclcpp::Node> make_sam2_trt_node(
    const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
