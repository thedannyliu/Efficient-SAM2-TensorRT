#include "sam2_trt_ros/sam2_trt_node.hpp"

#include "sam2_trt/tracker.hpp"
#include "sam2_trt_msgs/srv/add_object.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <array>
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

  explicit Sam2TrtNode(const rclcpp::NodeOptions& options)
      : Node("sam2_trt", options) {
    declare_parameter("model_id", "sam2.1-hiera-large");
    const auto bundle = declare_parameter("bundle_dir", "");
    const auto precision = declare_parameter("precision", "fp32");
    const auto image_topic = declare_parameter("image_topic", "/camera/camera/color/image_raw");
    const auto max_objects = declare_parameter("max_objects", 8);
    const auto track_concurrency = declare_parameter("track_concurrency", max_objects);
    const auto trace_path = declare_parameter("trace_path", "");
    const auto preview_width = declare_parameter("preview_width", 640);
    const auto preview_height = declare_parameter("preview_height", 360);
    declare_parameter("queue_policy", "latest");
    declare_parameter("enable_overlay", false);
    if (bundle.empty()) throw std::invalid_argument("bundle_dir parameter is required");
    tracker_ = std::make_unique<sam2_trt::Tracker>(
        bundle, precision, max_objects, track_concurrency);
    if (!trace_path.empty()) {
      const std::filesystem::path path(trace_path);
      if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
      trace_.open(path, std::ios::app);
      if (!trace_) throw std::runtime_error("cannot open trace_path: " + trace_path);
      RCLCPP_INFO(get_logger(), "Writing runtime trace to %s", trace_path.c_str());
    }

    if (preview_width < 1 || preview_height < 1)
      throw std::invalid_argument("preview dimensions must be positive");
    preview_width_ = preview_width;
    preview_height_ = preview_height;
    const auto image_qos = rclcpp::SensorDataQoS().keep_last(1);
    mask_publisher_ = create_publisher<sensor_msgs::msg::Image>(
        "/segmentation_mask", image_qos);
    object_mask_publisher_ = create_publisher<sensor_msgs::msg::Image>(
        "/sam/object_masks", image_qos);
    preview_publisher_ = create_publisher<sensor_msgs::msg::Image>(
        "/sam/preview", image_qos);
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
    preview_worker_ = std::jthread(
        [this](std::stop_token token) { preview_worker(token); });
    worker_ = std::jthread([this](std::stop_token token) { worker(token); });
  }

  ~Sam2TrtNode() override {
    worker_.request_stop();
    frame_ready_.notify_all();
    preview_worker_.request_stop();
    preview_ready_.notify_all();
  }

 private:
  struct PendingFrame {
    sensor_msgs::msg::Image::ConstSharedPtr message;
    SteadyClock::time_point arrival;
  };

  struct PreviewJob {
    sensor_msgs::msg::Image::ConstSharedPtr frame;
    std::vector<sam2_trt::ObjectMask> masks;
  };

  static double milliseconds(SteadyClock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
  }

  sensor_msgs::msg::Image make_preview(
      const sensor_msgs::msg::Image& frame,
      const std::vector<sam2_trt::ObjectMask>& masks) const {
    static constexpr std::array<std::array<std::uint8_t, 3>, 4> colors{{
        {{0, 255, 0}},
        {{255, 128, 0}},
        {{0, 128, 255}},
        {{255, 0, 255}},
    }};
    sensor_msgs::msg::Image preview;
    preview.header = frame.header;
    preview.height = static_cast<std::uint32_t>(preview_height_);
    preview.width = static_cast<std::uint32_t>(preview_width_);
    preview.encoding = sensor_msgs::image_encodings::RGB8;
    preview.is_bigendian = false;
    preview.step = static_cast<std::uint32_t>(preview_width_ * 3);
    preview.data.resize(
        static_cast<std::size_t>(preview.step) * preview.height);
    const bool input_rgb = frame.encoding == sensor_msgs::image_encodings::RGB8;
    for (int y = 0; y < preview_height_; ++y) {
      const int source_y = std::min(
          static_cast<int>(frame.height) - 1,
          y * static_cast<int>(frame.height) / preview_height_);
      for (int x = 0; x < preview_width_; ++x) {
        const int source_x = std::min(
            static_cast<int>(frame.width) - 1,
            x * static_cast<int>(frame.width) / preview_width_);
        const auto source_offset =
            static_cast<std::size_t>(source_y) * frame.step + source_x * 3;
        const auto output_offset =
            (static_cast<std::size_t>(y) * preview_width_ + x) * 3;
        for (int channel = 0; channel < 3; ++channel) {
          const int source_channel = input_rgb ? channel : 2 - channel;
          preview.data[output_offset + channel] =
              frame.data[source_offset + source_channel];
        }
        for (std::size_t index = 0; index < masks.size(); ++index) {
          const auto& mask = masks[index];
          const auto mask_x = std::min(mask.width - 1, source_x);
          const auto mask_y = std::min(mask.height - 1, source_y);
          if (mask.mono8[static_cast<std::size_t>(mask_y) * mask.width + mask_x] == 0)
            continue;
          const auto& color = colors[
              static_cast<std::size_t>(mask.object_id - 1) % colors.size()];
          for (int channel = 0; channel < 3; ++channel) {
            preview.data[output_offset + channel] = static_cast<std::uint8_t>(
                preview.data[output_offset + channel] * 0.55f +
                color[channel] * 0.45f);
          }
        }
      }
    }
    return preview;
  }

  void preview_worker(std::stop_token token) {
    while (!token.stop_requested()) {
      std::optional<PreviewJob> job;
      {
        std::unique_lock lock(preview_mutex_);
        preview_ready_.wait(lock, token, [this] {
          return latest_preview_.has_value();
        });
        if (token.stop_requested()) return;
        job = std::move(latest_preview_);
        latest_preview_.reset();
      }
      const auto start = SteadyClock::now();
      auto preview = make_preview(*job->frame, job->masks);
      last_preview_compose_ms_ = milliseconds(SteadyClock::now() - start);
      preview_publisher_->publish(std::move(preview));
    }
  }

  void enqueue_preview(
      sensor_msgs::msg::Image::ConstSharedPtr frame,
      std::vector<sam2_trt::ObjectMask> masks) {
    {
      std::lock_guard lock(preview_mutex_);
      if (latest_preview_) ++dropped_previews_;
      latest_preview_ = PreviewJob{std::move(frame), std::move(masks)};
    }
    preview_ready_.notify_one();
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
    const auto tracker_timings = tracker_->last_timings();
    const auto inference_end = SteadyClock::now();
    const auto mask_publish_start = inference_end;
    const bool publish_legacy_mask = mask_publisher_->get_subscription_count() > 0;
    const bool publish_object_masks =
        object_mask_publisher_->get_subscription_count() > 0;
    for (std::size_t index = 0;
         index < masks.size() && (publish_legacy_mask || publish_object_masks);
         ++index) {
      sensor_msgs::msg::Image message;
      message.header = frame->header;
      message.header.frame_id += "/sam_object_" + std::to_string(masks[index].object_id);
      message.height = masks[index].height;
      message.width = masks[index].width;
      message.encoding = sensor_msgs::image_encodings::MONO8;
      message.is_bigendian = false;
      message.step = masks[index].width;
      message.data = masks[index].mono8;
      if (index == 0 && publish_legacy_mask) mask_publisher_->publish(message);
      if (publish_object_masks) object_mask_publisher_->publish(std::move(message));
    }
    const auto metrics_time = SteadyClock::now();
    const double callback_total_ms = milliseconds(metrics_time - pending.arrival);
    const double worker_total_ms = milliseconds(metrics_time - worker_start);
    const auto dropped_total = dropped_frames_.load();
    const auto dropped = dropped_total - last_reported_dropped_frames_;
    last_reported_dropped_frames_ = dropped_total;

    std_msgs::msg::String result;
    std::ostringstream json;
    json << std::fixed << std::setprecision(3);
    json << "{\"stamp_ns\":" << rclcpp::Time(frame->header.stamp).nanoseconds()
         << ",\"frame_index\":" << processed_frames_++
         << ",\"source_width\":" << frame->width
         << ",\"source_height\":" << frame->height
         << ",\"objects\":[";
    for (std::size_t index = 0; index < masks.size(); ++index) {
      if (index) json << ',';
      json << masks[index].object_id;
    }
    json << "],\"queue_wait_ms\":" << queue_wait_ms
         << ",\"color_convert_ms\":" << milliseconds(color_end - color_start)
         << ",\"inference_ms\":" << milliseconds(inference_end - inference_start)
         << ",\"host_input_copy_ms\":" << tracker_timings.host_input_copy_ms
         << ",\"encoder_gpu_ms\":" << tracker_timings.encoder_gpu_ms
         << ",\"tail_gpu_ms\":" << tracker_timings.tail_gpu_ms
         << ",\"gpu_total_ms\":" << tracker_timings.gpu_total_ms
         << ",\"host_mask_copy_ms\":" << tracker_timings.host_mask_copy_ms
         << ",\"tracker_total_ms\":" << tracker_timings.total_ms
         << ",\"mask_publish_ms\":" << milliseconds(metrics_time - mask_publish_start)
         << ",\"preview_compose_ms\":" << last_preview_compose_ms_.load()
         << ",\"dropped_previews\":" << dropped_previews_.load()
         << ",\"callback_total_ms\":" << callback_total_ms
         << ",\"worker_total_ms\":" << worker_total_ms
         << ",\"frame_interval_ms\":" << frame_interval_ms
         << ",\"dropped\":" << dropped
         << ",\"dropped_frames\":" << dropped_total;
    if (worker_total_ms > 0.0)
      json << ",\"processing_capacity_fps\":" << 1000.0 / worker_total_ms;
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
    if (preview_publisher_->get_subscription_count() > 0)
      enqueue_preview(frame, std::move(masks));
  }

  std::unique_ptr<sam2_trt::Tracker> tracker_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr object_mask_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr preview_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr result_publisher_;
  rclcpp::Service<sam2_trt_msgs::srv::AddObject>::SharedPtr add_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
  std::mutex tracker_mutex_;
  std::mutex frame_mutex_;
  std::mutex preview_mutex_;
  std::condition_variable_any frame_ready_;
  std::condition_variable_any preview_ready_;
  std::optional<PendingFrame> latest_;
  std::optional<PreviewJob> latest_preview_;
  std::atomic<std::uint64_t> dropped_frames_{0};
  std::atomic<std::uint64_t> dropped_previews_{0};
  std::atomic<double> last_preview_compose_ms_{0.0};
  std::uint64_t last_reported_dropped_frames_{0};
  std::uint64_t processed_frames_{0};
  std::optional<SteadyClock::time_point> previous_worker_start_;
  int preview_width_{};
  int preview_height_{};
  std::ofstream trace_;
  std::jthread preview_worker_;
  std::jthread worker_;
};

std::shared_ptr<rclcpp::Node> make_sam2_trt_node(
    const rclcpp::NodeOptions& options) {
  return std::make_shared<Sam2TrtNode>(options);
}

RCLCPP_COMPONENTS_REGISTER_NODE(Sam2TrtNode)
