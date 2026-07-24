#include "sam2_trt/tracker.hpp"
#include "sam2_trt_msgs/srv/add_object.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <condition_variable>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

class Sam2TrtNode final : public rclcpp::Node {
 public:
  using SteadyClock = std::chrono::steady_clock;

  Sam2TrtNode() : Node("sam2_trt") {
    declare_parameter("model_id", "sam2.1-hiera-large");
    const auto bundle = declare_parameter("bundle_dir", "");
    const auto precision = declare_parameter("precision", "fp32");
    const auto image_topic = declare_parameter("image_topic", "/camera/camera/color/image_raw");
    const auto max_objects = declare_parameter("max_objects", 8);
    const auto trace_path = declare_parameter("trace_path", "");
    declare_parameter("queue_policy", "latest");
    declare_parameter("enable_overlay", false);
    if (bundle.empty()) throw std::invalid_argument("bundle_dir parameter is required");
    tracker_ = std::make_unique<sam2_trt::Tracker>(bundle, precision, max_objects);
    if (!trace_path.empty()) {
      const std::filesystem::path path(trace_path);
      if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
      trace_.open(path, std::ios::app);
      if (!trace_) throw std::runtime_error("cannot open trace_path: " + trace_path);
      RCLCPP_INFO(get_logger(), "Writing runtime trace to %s", trace_path.c_str());
    }

    mask_publisher_ = create_publisher<sensor_msgs::msg::Image>("/segmentation_mask", 1);
    object_mask_publisher_ = create_publisher<sensor_msgs::msg::Image>("/sam/object_masks", 8);
    result_publisher_ = create_publisher<std_msgs::msg::String>("/sam/result_json", 10);
    subscription_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic, rclcpp::SensorDataQoS().keep_last(1),
        [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
          const auto arrival = SteadyClock::now();
          {
            std::lock_guard lock(frame_mutex_);
            if (latest_) ++dropped_frames_;
            latest_ = PendingFrame{std::move(message), arrival};
          }
          frame_ready_.notify_one();
        });
    add_service_ = create_service<sam2_trt_msgs::srv::AddObject>(
        "/sam/add_object",
        [this](sam2_trt_msgs::srv::AddObject::Request::SharedPtr request,
               sam2_trt_msgs::srv::AddObject::Response::SharedPtr response) {
          try {
            sam2_trt::Prompt prompt;
            prompt.kind = request->kind == sam2_trt_msgs::srv::AddObject::Request::BOX
                              ? sam2_trt::PromptKind::Box : sam2_trt::PromptKind::Point;
            prompt.x0 = request->x0; prompt.y0 = request->y0;
            prompt.x1 = request->x1; prompt.y1 = request->y1;
            std::lock_guard lock(tracker_mutex_);
            response->object_id = tracker_->add_object(prompt);
            response->success = true;
          } catch (const std::exception& error) {
            response->success = false;
            response->message = error.what();
          }
        });
    reset_service_ = create_service<std_srvs::srv::Trigger>(
        "/sam/reset", [this](std_srvs::srv::Trigger::Request::SharedPtr,
                            std_srvs::srv::Trigger::Response::SharedPtr response) {
          std::lock_guard lock(tracker_mutex_);
          tracker_->reset(); response->success = true;
        });
    worker_ = std::jthread([this](std::stop_token token) { worker(token); });
  }

  ~Sam2TrtNode() override {
    worker_.request_stop();
    frame_ready_.notify_all();
  }

 private:
  struct PendingFrame {
    sensor_msgs::msg::Image::ConstSharedPtr message;
    SteadyClock::time_point arrival;
  };

  static double milliseconds(SteadyClock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
  }

  void worker(std::stop_token token) {
    while (!token.stop_requested()) {
      std::optional<PendingFrame> pending;
      {
        std::unique_lock lock(frame_mutex_);
        frame_ready_.wait(lock, token, [this] { return latest_.has_value(); });
        if (token.stop_requested()) return;
        pending = std::move(latest_);
        latest_.reset();
      }
      try { process(std::move(*pending)); }
      catch (const std::exception& error) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "inference failed: %s", error.what());
      }
    }
  }

  void process(PendingFrame pending) {
    const auto& frame = pending.message;
    const auto worker_start = SteadyClock::now();
    const double queue_wait_ms = milliseconds(worker_start - pending.arrival);
    double frame_interval_ms = 0.0;
    if (previous_worker_start_) frame_interval_ms = milliseconds(worker_start - *previous_worker_start_);
    previous_worker_start_ = worker_start;

    const auto color_start = SteadyClock::now();
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
    const auto color_end = SteadyClock::now();
    const auto inference_start = color_end;
    std::vector<sam2_trt::ObjectMask> masks;
    {
      std::lock_guard lock(tracker_mutex_);
      masks = tracker_->process_rgb8(rgb, frame->width, frame->height, stride);
    }
    const auto inference_end = SteadyClock::now();
    const auto mask_publish_start = inference_end;
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
    const auto metrics_time = SteadyClock::now();
    const double callback_total_ms = milliseconds(metrics_time - pending.arrival);
    const auto dropped_total = dropped_frames_.load();
    const auto dropped = dropped_total - last_reported_dropped_frames_;
    last_reported_dropped_frames_ = dropped_total;

    std_msgs::msg::String result;
    std::ostringstream json;
    json << std::fixed << std::setprecision(3);
    json << "{\"stamp_ns\":" << rclcpp::Time(frame->header.stamp).nanoseconds()
         << ",\"frame_index\":" << processed_frames_++
         << ",\"objects\":[";
    for (std::size_t index = 0; index < masks.size(); ++index) {
      if (index) json << ',';
      json << masks[index].object_id;
    }
    json << "],\"queue_wait_ms\":" << queue_wait_ms
         << ",\"color_convert_ms\":" << milliseconds(color_end - color_start)
         << ",\"inference_ms\":" << milliseconds(inference_end - inference_start)
         << ",\"mask_publish_ms\":" << milliseconds(metrics_time - mask_publish_start)
         << ",\"callback_total_ms\":" << callback_total_ms
         << ",\"frame_interval_ms\":" << frame_interval_ms
         << ",\"dropped\":" << dropped
         << ",\"dropped_frames\":" << dropped_total;
    if (callback_total_ms > 0.0)
      json << ",\"processing_capacity_fps\":" << 1000.0 / callback_total_ms;
    if (frame_interval_ms > 0.0) {
      const double processed_fps = 1000.0 / frame_interval_ms;
      json << ",\"processed_fps\":" << processed_fps
           << ",\"tracking_fps\":" << processed_fps;
    }
    const auto stamp_ns = rclcpp::Time(frame->header.stamp).nanoseconds();
    if (stamp_ns > 0) {
      const auto source_age_ns = (get_clock()->now() - rclcpp::Time(frame->header.stamp)).nanoseconds();
      if (source_age_ns >= 0) {
        const double source_age_ms = static_cast<double>(source_age_ns) / 1.0e6;
        json << ",\"source_age_ms\":" << source_age_ms
             << ",\"end_to_end_ms\":" << source_age_ms;
      }
    }
    json << '}';
    result.data = json.str();
    result_publisher_->publish(result);
    if (trace_) trace_ << result.data << '\n';
  }

  std::unique_ptr<sam2_trt::Tracker> tracker_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr object_mask_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr result_publisher_;
  rclcpp::Service<sam2_trt_msgs::srv::AddObject>::SharedPtr add_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
  std::mutex tracker_mutex_;
  std::mutex frame_mutex_;
  std::condition_variable_any frame_ready_;
  std::optional<PendingFrame> latest_;
  std::atomic<std::uint64_t> dropped_frames_{0};
  std::uint64_t last_reported_dropped_frames_{0};
  std::uint64_t processed_frames_{0};
  std::optional<SteadyClock::time_point> previous_worker_start_;
  std::ofstream trace_;
  std::jthread worker_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Sam2TrtNode>());
  rclcpp::shutdown();
  return 0;
}
