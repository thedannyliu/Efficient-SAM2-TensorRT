#include "sam2_trt/tracker.hpp"
#include "sam2_trt_msgs/srv/add_object.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <condition_variable>
#include <atomic>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

class Sam2TrtNode final : public rclcpp::Node {
 public:
  Sam2TrtNode() : Node("sam2_trt") {
    declare_parameter("model_id", "sam2.1-hiera-large");
    const auto bundle = declare_parameter("bundle_dir", "");
    const auto precision = declare_parameter("precision", "fp32");
    const auto image_topic = declare_parameter("image_topic", "/camera/camera/color/image_raw");
    const auto max_objects = declare_parameter("max_objects", 8);
    declare_parameter("queue_policy", "latest");
    declare_parameter("enable_overlay", false);
    if (bundle.empty()) throw std::invalid_argument("bundle_dir parameter is required");
    tracker_ = std::make_unique<sam2_trt::Tracker>(bundle, precision, max_objects);

    mask_publisher_ = create_publisher<sensor_msgs::msg::Image>("/segmentation_mask", 1);
    object_mask_publisher_ = create_publisher<sensor_msgs::msg::Image>("/sam/object_masks", 8);
    result_publisher_ = create_publisher<std_msgs::msg::String>("/sam/result_json", 10);
    subscription_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic, rclcpp::SensorDataQoS().keep_last(1),
        [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
          {
            std::lock_guard lock(frame_mutex_);
            if (latest_) ++dropped_frames_;
            latest_ = std::move(message);
          }
          frame_ready_.notify_one();
        });
    add_service_ = create_service<sam2_trt_msgs::srv::AddObject>(
        "/sam/add_object",
        [this](const auto request, auto response) {
          try {
            sam2_trt::Prompt prompt;
            prompt.kind = request->kind == sam2_trt_msgs::srv::AddObject::Request::BOX
                              ? sam2_trt::PromptKind::Box : sam2_trt::PromptKind::Point;
            prompt.x0 = request->x0; prompt.y0 = request->y0;
            prompt.x1 = request->x1; prompt.y1 = request->y1;
            response->object_id = tracker_->add_object(prompt);
            response->success = true;
          } catch (const std::exception& error) {
            response->success = false;
            response->message = error.what();
          }
        });
    reset_service_ = create_service<std_srvs::srv::Trigger>(
        "/sam/reset", [this](const auto, auto response) {
          tracker_->reset(); response->success = true;
        });
    worker_ = std::jthread([this](std::stop_token token) { worker(token); });
  }

  ~Sam2TrtNode() override {
    worker_.request_stop();
    frame_ready_.notify_all();
  }

 private:
  void worker(std::stop_token token) {
    while (!token.stop_requested()) {
      sensor_msgs::msg::Image::ConstSharedPtr frame;
      {
        std::unique_lock lock(frame_mutex_);
        frame_ready_.wait(lock, token, [this] { return static_cast<bool>(latest_); });
        if (token.stop_requested()) return;
        frame = std::move(latest_);
      }
      try { process(frame); }
      catch (const std::exception& error) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "inference failed: %s", error.what());
      }
    }
  }

  void process(const sensor_msgs::msg::Image::ConstSharedPtr& frame) {
    const std::uint8_t* rgb = frame->data.data();
    std::vector<std::uint8_t> converted;
    std::size_t stride = frame->step;
    if (frame->encoding == sensor_msgs::image_encodings::BGR8) {
      converted.resize(static_cast<std::size_t>(frame->width) * frame->height * 3);
      for (std::size_t y = 0; y < frame->height; ++y)
        for (std::size_t x = 0; x < frame->width; ++x)
          for (int channel = 0; channel < 3; ++channel)
            converted[(y * frame->width + x) * 3 + channel] =
                frame->data[y * frame->step + x * 3 + (2 - channel)];
      rgb = converted.data();
      stride = static_cast<std::size_t>(frame->width) * 3;
    } else if (frame->encoding != sensor_msgs::image_encodings::RGB8) {
      throw std::invalid_argument("camera image must use rgb8 or bgr8 encoding");
    }
    auto masks = tracker_->process_rgb8(rgb, frame->width, frame->height, stride);
    for (std::size_t index = 0; index < masks.size(); ++index) {
      sensor_msgs::msg::Image message;
      message.header = frame->header;
      message.header.frame_id += "/sam_object_" + std::to_string(masks[index].object_id);
      message.height = masks[index].height;
      message.width = masks[index].width;
      message.encoding = sensor_msgs::image_encodings::MONO8;
      message.is_bigendian = false;
      message.step = masks[index].width;
      message.data = std::move(masks[index].mono8);
      if (index == 0) mask_publisher_->publish(message);
      object_mask_publisher_->publish(std::move(message));
    }
    std_msgs::msg::String result;
    std::ostringstream json;
    json << "{\"stamp_ns\":" << rclcpp::Time(frame->header.stamp).nanoseconds()
         << ",\"objects\":[";
    for (std::size_t index = 0; index < masks.size(); ++index) {
      if (index) json << ',';
      json << masks[index].object_id;
    }
    json << "],\"dropped_frames\":" << dropped_frames_.load() << '}';
    result.data = json.str();
    result_publisher_->publish(result);
  }

  std::unique_ptr<sam2_trt::Tracker> tracker_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr object_mask_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr result_publisher_;
  rclcpp::Service<sam2_trt_msgs::srv::AddObject>::SharedPtr add_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
  std::mutex frame_mutex_;
  std::condition_variable_any frame_ready_;
  sensor_msgs::msg::Image::ConstSharedPtr latest_;
  std::atomic<std::uint64_t> dropped_frames_{0};
  std::jthread worker_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Sam2TrtNode>());
  rclcpp::shutdown();
  return 0;
}
