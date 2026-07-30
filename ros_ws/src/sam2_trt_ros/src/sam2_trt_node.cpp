#include "sam2_trt_ros/sam2_trt_node.hpp"

#include "sam2_trt/tracker.hpp"
#include "sam2_trt_msgs/srv/add_mask.hpp"
#include "sam2_trt_msgs/srv/add_object.hpp"
#include "sam2_trt_msgs/srv/switch_model.hpp"

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
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

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

class Sam2TrtNode final : public rclcpp::Node {
 public:
  using SteadyClock = std::chrono::steady_clock;

  explicit Sam2TrtNode(const rclcpp::NodeOptions& options)
      : Node("sam2_trt", options) {
    model_id_ = declare_parameter("model_id", "sam2.1-hiera-large");
    const auto bundle = declare_parameter("bundle_dir", "");
    precision_ = declare_parameter("precision", "fp32");
    const auto image_topic = declare_parameter("image_topic", "/camera/camera/color/image_raw");
    shared_memory_path_ = declare_parameter("shared_memory_path", "");
    const auto shared_memory_poll_hz =
        declare_parameter("shared_memory_poll_hz", 240.0);
    if (shared_memory_poll_hz <= 0.0)
      throw std::invalid_argument("shared_memory_poll_hz must be positive");
    shared_memory_poll_period_ = std::chrono::microseconds(
        static_cast<std::int64_t>(1.0e6 / shared_memory_poll_hz));
    max_objects_ = declare_parameter("max_objects", 8);
    track_concurrency_ = declare_parameter("track_concurrency", max_objects_);
    track_bucket_size_ = declare_parameter("track_bucket_size", 1);
    track_bucket_min_objects_ =
        declare_parameter("track_bucket_min_objects", 4);
    pipeline_overlap_ = declare_parameter("pipeline_overlap", false);
    pipeline_overlap_max_objects_ =
        declare_parameter("pipeline_overlap_max_objects", 1);
    if (pipeline_overlap_max_objects_ < 1 ||
        pipeline_overlap_max_objects_ > max_objects_)
      throw std::invalid_argument(
          "pipeline_overlap_max_objects must be in [1, max_objects]");
    const auto trace_path = declare_parameter("trace_path", "");
    const auto preview_width = declare_parameter("preview_width", 640);
    const auto preview_height = declare_parameter("preview_height", 360);
    declare_parameter("queue_policy", "latest");
    declare_parameter("enable_overlay", false);
    if (bundle.empty()) throw std::invalid_argument("bundle_dir parameter is required");
    bundle_dir_ = bundle;
    tracker_ = std::make_unique<sam2_trt::Tracker>(
        bundle_dir_, precision_, max_objects_, track_concurrency_,
        track_bucket_size_, track_bucket_min_objects_);
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
    preview_label_publisher_ = create_publisher<sensor_msgs::msg::Image>(
        "/sam/preview_labels", image_qos);
    result_publisher_ = create_publisher<std_msgs::msg::String>("/sam/result_json", 10);
    subscription_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic, rclcpp::SensorDataQoS().keep_last(1),
        [this](sensor_msgs::msg::Image::ConstSharedPtr message) {
          enqueue_frame(std::move(message), SteadyClock::now(), 0.0, false);
        });
    add_mask_service_ = create_service<sam2_trt_msgs::srv::AddMask>(
        "/sam/add_mask",
        [this](sam2_trt_msgs::srv::AddMask::Request::SharedPtr request,
               sam2_trt_msgs::srv::AddMask::Response::SharedPtr response) {
          try {
            const auto& mask = request->mask;
            if (mask.encoding != sensor_msgs::image_encodings::MONO8)
              throw std::invalid_argument("mask prompt must use mono8 encoding");
            if (mask.width < 1 || mask.height < 1 || mask.step < mask.width ||
                mask.data.size() <
                    static_cast<std::size_t>(mask.step) * mask.height)
              throw std::invalid_argument("mask prompt payload is incomplete");
            sam2_trt::Prompt prompt;
            prompt.kind = sam2_trt::PromptKind::Mask;
            prompt.mask_width = static_cast<int>(mask.width);
            prompt.mask_height = static_cast<int>(mask.height);
            prompt.mask_stride = mask.step;
            prompt.mask = mask.data;
            std::lock_guard lock(tracker_mutex_);
            response->object_id = tracker_->add_object(prompt);
            ++object_count_;
            response->success = true;
          } catch (const std::exception& error) {
            response->success = false;
            response->message = error.what();
          }
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
            ++object_count_;
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
          tracker_->reset();
          pipeline_pending_.reset();
          pipeline_overlap_active_ = false;
          object_count_ = 0;
          response->success = true;
        });
    switch_model_service_ = create_service<sam2_trt_msgs::srv::SwitchModel>(
        "/sam/switch_model",
        [this](
            sam2_trt_msgs::srv::SwitchModel::Request::SharedPtr request,
            sam2_trt_msgs::srv::SwitchModel::Response::SharedPtr response) {
          switch_model(*request, *response);
        });
    preview_worker_ = std::jthread(
        [this](std::stop_token token) { preview_worker(token); });
    worker_ = std::jthread([this](std::stop_token token) { worker(token); });
    if (!shared_memory_path_.empty()) {
      shared_frame_worker_ = std::jthread(
          [this](std::stop_token token) { shared_frame_worker(token); });
      RCLCPP_INFO(
          get_logger(), "Reading latest camera frame from %s",
          shared_memory_path_.c_str());
    }
  }

  ~Sam2TrtNode() override {
    shared_frame_worker_.request_stop();
    worker_.request_stop();
    frame_ready_.notify_all();
    preview_worker_.request_stop();
    preview_ready_.notify_all();
  }

 private:
  struct PendingFrame {
    sensor_msgs::msg::Image::ConstSharedPtr message;
    SteadyClock::time_point arrival;
    double input_transport_ms{};
    bool shared_memory{};
  };

#pragma pack(push, 1)
  struct SharedFrameHeader {
    char magic[8];
    std::uint64_t sequence;
    std::uint64_t stamp_ns;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t stride;
    std::uint32_t payload_bytes;
  };
#pragma pack(pop)

  static_assert(sizeof(SharedFrameHeader) == 40);

  struct PreviewJob {
    sensor_msgs::msg::Image::ConstSharedPtr frame;
    std::vector<sam2_trt::ObjectMask> masks;
  };

  static double milliseconds(SteadyClock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
  }

  void enqueue_frame(
      sensor_msgs::msg::Image::ConstSharedPtr message,
      SteadyClock::time_point arrival,
      double input_transport_ms,
      bool shared_memory) {
    {
      std::lock_guard lock(frame_mutex_);
      if (latest_) ++dropped_frames_;
      latest_ = PendingFrame{
          std::move(message), arrival, input_transport_ms, shared_memory};
    }
    frame_ready_.notify_one();
  }

  static bool read_exact(
      int file_descriptor, void* destination, std::size_t bytes,
      off_t offset) {
    auto* output = static_cast<std::uint8_t*>(destination);
    std::size_t completed = 0;
    while (completed < bytes) {
      const auto count = ::pread(
          file_descriptor, output + completed, bytes - completed,
          offset + static_cast<off_t>(completed));
      if (count <= 0) return false;
      completed += static_cast<std::size_t>(count);
    }
    return true;
  }

  void shared_frame_worker(std::stop_token token) {
    int file_descriptor = -1;
    std::uint64_t last_sequence = 0;
    while (!token.stop_requested()) {
      if (file_descriptor < 0) {
        file_descriptor = ::open(shared_memory_path_.c_str(), O_RDONLY);
        if (file_descriptor < 0) {
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          continue;
        }
      }

      const auto read_start = SteadyClock::now();
      if (::flock(file_descriptor, LOCK_SH) != 0) {
        ::close(file_descriptor);
        file_descriptor = -1;
        continue;
      }
      SharedFrameHeader header{};
      bool valid = read_exact(
          file_descriptor, &header, sizeof(header), 0);
      valid = valid &&
          std::memcmp(header.magic, "SAM2RGB1", sizeof(header.magic)) == 0 &&
          header.sequence != last_sequence &&
          header.width > 0 && header.height > 0 &&
          header.stride >= header.width * 3 &&
          header.payload_bytes ==
              static_cast<std::uint64_t>(header.stride) * header.height &&
          header.payload_bytes <= 64U * 1024U * 1024U;

      sensor_msgs::msg::Image::SharedPtr message;
      if (valid) {
        message = std::make_shared<sensor_msgs::msg::Image>();
        message->header.stamp.sec = static_cast<std::int32_t>(
            header.stamp_ns / 1'000'000'000ULL);
        message->header.stamp.nanosec = static_cast<std::uint32_t>(
            header.stamp_ns % 1'000'000'000ULL);
        message->header.frame_id = "instinctsam_shared";
        message->height = header.height;
        message->width = header.width;
        message->encoding = sensor_msgs::image_encodings::RGB8;
        message->is_bigendian = false;
        message->step = header.stride;
        message->data.resize(header.payload_bytes);
        valid = read_exact(
            file_descriptor, message->data.data(), message->data.size(),
            static_cast<off_t>(sizeof(header)));
      }
      ::flock(file_descriptor, LOCK_UN);

      if (valid) {
        last_sequence = header.sequence;
        const double transport_ms =
            milliseconds(SteadyClock::now() - read_start);
        enqueue_frame(
            std::move(message), read_start, transport_ms, true);
      } else {
        std::this_thread::sleep_for(shared_memory_poll_period_);
      }
    }
    if (file_descriptor >= 0) ::close(file_descriptor);
  }

  void switch_model(
      const sam2_trt_msgs::srv::SwitchModel::Request& request,
      sam2_trt_msgs::srv::SwitchModel::Response& response) {
    if (request.model_id.empty() || request.bundle_dir.empty() ||
        request.precision.empty()) {
      response.message = "model_id, bundle_dir, and precision are required";
      response.active_model_id = model_id_;
      return;
    }
    {
      std::lock_guard lock(tracker_mutex_);
      if (request.model_id == model_id_ &&
          request.bundle_dir == bundle_dir_ &&
          request.precision == precision_) {
        response.success = true;
        response.active_model_id = model_id_;
        response.message = model_id_ + " is already active";
        return;
      }
    }

    const auto start = SteadyClock::now();
    try {
      auto replacement = std::make_unique<sam2_trt::Tracker>(
          request.bundle_dir, request.precision, max_objects_,
          track_concurrency_, track_bucket_size_,
          track_bucket_min_objects_);
      {
        std::lock_guard lock(tracker_mutex_);
        tracker_.swap(replacement);
        model_id_ = request.model_id;
        bundle_dir_ = request.bundle_dir;
        precision_ = request.precision;
        pipeline_pending_.reset();
        pipeline_overlap_active_ = false;
        object_count_ = 0;
        replacement.reset();
      }
      response.success = true;
      response.active_model_id = model_id_;
      response.load_ms = milliseconds(SteadyClock::now() - start);
      response.message = "switched to " + model_id_;
      RCLCPP_INFO(
          get_logger(), "Switched to %s in %.1f ms", model_id_.c_str(),
          response.load_ms);
    } catch (const std::exception& error) {
      response.active_model_id = model_id_;
      response.load_ms = milliseconds(SteadyClock::now() - start);
      response.message = error.what();
      RCLCPP_ERROR(
          get_logger(), "Model switch to %s failed after %.1f ms: %s",
          request.model_id.c_str(), response.load_ms, error.what());
    }
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

  sensor_msgs::msg::Image make_preview_labels(
      const sensor_msgs::msg::Image& frame,
      const std::vector<sam2_trt::ObjectMask>& masks) const {
    sensor_msgs::msg::Image labels;
    labels.header = frame.header;
    labels.height = static_cast<std::uint32_t>(preview_height_);
    labels.width = static_cast<std::uint32_t>(preview_width_);
    labels.encoding = sensor_msgs::image_encodings::MONO8;
    labels.is_bigendian = false;
    labels.step = static_cast<std::uint32_t>(preview_width_);
    labels.data.assign(
        static_cast<std::size_t>(labels.step) * labels.height, 0);
    for (int y = 0; y < preview_height_; ++y) {
      const int source_y = std::min(
          static_cast<int>(frame.height) - 1,
          y * static_cast<int>(frame.height) / preview_height_);
      for (int x = 0; x < preview_width_; ++x) {
        const int source_x = std::min(
            static_cast<int>(frame.width) - 1,
            x * static_cast<int>(frame.width) / preview_width_);
        for (const auto& mask : masks) {
          const auto mask_x = std::min(mask.width - 1, source_x);
          const auto mask_y = std::min(mask.height - 1, source_y);
          if (mask.mono8[
                  static_cast<std::size_t>(mask_y) * mask.width + mask_x] != 0)
            labels.data[
                static_cast<std::size_t>(y) * preview_width_ + x] =
                static_cast<std::uint8_t>(mask.object_id);
        }
      }
    }
    return labels;
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
      if (preview_publisher_->get_subscription_count() > 0) {
        auto preview = make_preview(*job->frame, job->masks);
        preview_publisher_->publish(std::move(preview));
      }
      if (preview_label_publisher_->get_subscription_count() > 0) {
        auto labels = make_preview_labels(*job->frame, job->masks);
        preview_label_publisher_->publish(std::move(labels));
      }
      last_preview_compose_ms_ = milliseconds(SteadyClock::now() - start);
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
    const auto& input_frame = pending.message;
    const auto worker_start = SteadyClock::now();
    const double queue_wait_ms = milliseconds(worker_start - pending.arrival);
    double frame_interval_ms = 0.0;
    if (previous_worker_start_) frame_interval_ms = milliseconds(worker_start - *previous_worker_start_);
    previous_worker_start_ = worker_start;

    const auto color_start = SteadyClock::now();
    const std::uint8_t* rgb = input_frame->data.data();
    std::vector<std::uint8_t> converted;
    std::size_t stride = input_frame->step;
    if (input_frame->encoding == sensor_msgs::image_encodings::BGR8) {
      converted.resize(
          static_cast<std::size_t>(input_frame->width) *
          input_frame->height * 3);
      for (std::size_t y = 0; y < input_frame->height; ++y)
        for (std::size_t x = 0; x < input_frame->width; ++x)
          for (int channel = 0; channel < 3; ++channel)
            converted[(y * input_frame->width + x) * 3 + channel] =
                input_frame->data[
                    y * input_frame->step + x * 3 + (2 - channel)];
      rgb = converted.data();
      stride = static_cast<std::size_t>(input_frame->width) * 3;
    } else if (input_frame->encoding != sensor_msgs::image_encodings::RGB8) {
      throw std::invalid_argument("camera image must use rgb8 or bgr8 encoding");
    }
    const auto color_end = SteadyClock::now();
    const auto inference_start = color_end;
    std::vector<sam2_trt::ObjectMask> masks;
    PendingFrame output_pending;
    sam2_trt::TrackerTimings tracker_timings;
    std::string active_model_id;
    bool active_overlap = false;
    {
      std::lock_guard lock(tracker_mutex_);
      const bool routed_overlap =
          pipeline_overlap_ && object_count_ > 0 &&
          object_count_ <= pipeline_overlap_max_objects_;
      if (routed_overlap != pipeline_overlap_active_) {
        tracker_->discard_pipelined_frame();
        pipeline_pending_.reset();
        pipeline_overlap_active_ = routed_overlap;
      }
      active_overlap = pipeline_overlap_active_;
      if (active_overlap) {
        auto result = tracker_->process_pipelined_rgb8(
            rgb, input_frame->width, input_frame->height, stride);
        if (!result) {
          pipeline_pending_ = std::move(pending);
          return;
        }
        if (!pipeline_pending_)
          throw std::runtime_error("pipelined tracker lost its source frame");
        masks = std::move(*result);
        output_pending = std::move(*pipeline_pending_);
        pipeline_pending_ = std::move(pending);
      } else {
        masks = tracker_->process_rgb8(
            rgb, input_frame->width, input_frame->height, stride);
        output_pending = std::move(pending);
      }
      tracker_timings = tracker_->last_timings();
      active_model_id = model_id_;
    }
    const auto& frame = output_pending.message;
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
    const double callback_total_ms = milliseconds(
        metrics_time - output_pending.arrival);
    const double worker_total_ms = milliseconds(metrics_time - worker_start);
    const auto dropped_total = dropped_frames_.load();
    const auto dropped = dropped_total - last_reported_dropped_frames_;
    last_reported_dropped_frames_ = dropped_total;

    std_msgs::msg::String result;
    std::ostringstream json;
    json << std::fixed << std::setprecision(3);
    json << "{\"stamp_ns\":" << rclcpp::Time(frame->header.stamp).nanoseconds()
         << ",\"frame_index\":" << processed_frames_++
         << ",\"model_id\":\"" << active_model_id << '"'
         << ",\"source_width\":" << frame->width
         << ",\"source_height\":" << frame->height
         << ",\"objects\":[";
    for (std::size_t index = 0; index < masks.size(); ++index) {
      if (index) json << ',';
      json << masks[index].object_id;
    }
    json << "],\"queue_wait_ms\":" << queue_wait_ms
         << ",\"input_transport\":\""
         << (output_pending.shared_memory ? "shared_memory" : "ros_image")
         << '"'
         << ",\"input_transport_ms\":" << output_pending.input_transport_ms
         << ",\"pipeline_overlap\":"
         << (active_overlap ? "true" : "false")
         << ",\"pipeline_overlap_configured\":"
         << (pipeline_overlap_ ? "true" : "false")
         << ",\"pipeline_overlap_max_objects\":"
         << pipeline_overlap_max_objects_
         << ",\"pipeline_delay_frames\":"
         << (active_overlap ? 1 : 0)
         << ",\"track_bucket_active\":"
         << (track_bucket_size_ > 1 &&
                     object_count_ >= track_bucket_min_objects_
                 ? "true"
                 : "false")
         << ",\"track_bucket_size\":" << track_bucket_size_
         << ",\"track_bucket_min_objects\":"
         << track_bucket_min_objects_
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
    if (preview_publisher_->get_subscription_count() > 0 ||
        preview_label_publisher_->get_subscription_count() > 0)
      enqueue_preview(frame, std::move(masks));
  }

  std::unique_ptr<sam2_trt::Tracker> tracker_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr object_mask_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr preview_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr
      preview_label_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr result_publisher_;
  rclcpp::Service<sam2_trt_msgs::srv::AddObject>::SharedPtr add_service_;
  rclcpp::Service<sam2_trt_msgs::srv::AddMask>::SharedPtr add_mask_service_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_service_;
  rclcpp::Service<sam2_trt_msgs::srv::SwitchModel>::SharedPtr
      switch_model_service_;
  std::mutex tracker_mutex_;
  std::mutex frame_mutex_;
  std::mutex preview_mutex_;
  std::condition_variable_any frame_ready_;
  std::condition_variable_any preview_ready_;
  std::optional<PendingFrame> latest_;
  std::optional<PendingFrame> pipeline_pending_;
  std::optional<PreviewJob> latest_preview_;
  std::atomic<std::uint64_t> dropped_frames_{0};
  std::atomic<std::uint64_t> dropped_previews_{0};
  std::atomic<double> last_preview_compose_ms_{0.0};
  std::uint64_t last_reported_dropped_frames_{0};
  std::uint64_t processed_frames_{0};
  std::optional<SteadyClock::time_point> previous_worker_start_;
  int preview_width_{};
  int preview_height_{};
  int max_objects_{};
  int track_concurrency_{};
  int track_bucket_size_{};
  int track_bucket_min_objects_{};
  bool pipeline_overlap_{};
  int pipeline_overlap_max_objects_{};
  bool pipeline_overlap_active_{};
  int object_count_{};
  std::string model_id_;
  std::string bundle_dir_;
  std::string precision_;
  std::string shared_memory_path_;
  std::chrono::microseconds shared_memory_poll_period_{};
  std::ofstream trace_;
  std::jthread shared_frame_worker_;
  std::jthread preview_worker_;
  std::jthread worker_;
};

std::shared_ptr<rclcpp::Node> make_sam2_trt_node(
    const rclcpp::NodeOptions& options) {
  return std::make_shared<Sam2TrtNode>(options);
}

RCLCPP_COMPONENTS_REGISTER_NODE(Sam2TrtNode)
