#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from collections import deque
from threading import Thread
from time import perf_counter, sleep

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from sam2_trt.interactive import (
    all_masks_ready,
    display_to_image_point,
    drag_to_box,
    event_rate_hz,
)
from sam2_trt_msgs.srv import AddObject


_OBJECT_ID = re.compile(r"/sam_object_(\d+)$")
_COLORS = ((0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255))


class InteractiveViewer(Node):
    def __init__(self) -> None:
        super().__init__("sam2_trt_interactive_viewer")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("preview_topic", "/sam/preview")
        self.declare_parameter("mask_topic", "/sam/object_masks")
        self.declare_parameter("result_topic", "/sam/result_json")
        self.declare_parameter("window_name", "SAM2 TensorRT tracking")
        self.declare_parameter("display_fps", 30.0)
        self.declare_parameter("display_scale", 1.0)
        self.declare_parameter("display_max_width", 1280)
        self.declare_parameter("box_drag_min_pixels", 5.0)
        self.declare_parameter("replace_on_prompt", True)
        self.declare_parameter("draw_contours", False)
        self.declare_parameter("use_preview", True)
        self.declare_parameter("source_width", 1280)
        self.declare_parameter("source_height", 720)

        self.bridge = CvBridge()
        self.window_name = str(self.get_parameter("window_name").value)
        self.display_scale = float(self.get_parameter("display_scale").value)
        self.display_max_width = int(self.get_parameter("display_max_width").value)
        self.box_drag_min_pixels = float(self.get_parameter("box_drag_min_pixels").value)
        self.replace_on_prompt = bool(self.get_parameter("replace_on_prompt").value)
        self.draw_contours = bool(self.get_parameter("draw_contours").value)
        self.use_preview = bool(self.get_parameter("use_preview").value)
        self.source_width = int(self.get_parameter("source_width").value)
        self.source_height = int(self.get_parameter("source_height").value)
        self.current_scale = 1.0
        self.frames: dict[int, np.ndarray] = {}
        self.frame_order: deque[int] = deque()
        self.masks: dict[int, dict[int, np.ndarray]] = {}
        self.results: dict[int, dict[str, object]] = {}
        self.latest_frame: np.ndarray | None = None
        self.latest_overlay: np.ndarray | None = None
        self.latest_overlay_stamp = 0
        self.latest_result: dict[str, object] = {}
        self.result_times: deque[float] = deque(maxlen=120)
        self.tracker_stamps: deque[float] = deque(maxlen=120)
        self.present_times: deque[float] = deque(maxlen=120)
        self.last_presented_stamp = 0
        self.compose_ms = 0.0
        self.display_ms = 0.0
        self.last_metrics_log = perf_counter()
        self.color_layers: dict[tuple[int, int, int], np.ndarray] = {}
        self.blend_buffer: np.ndarray | None = None
        self.drag_start: tuple[float, float] | None = None
        self.drag_current: tuple[float, float] | None = None
        self.prompt_marker: tuple[str, tuple[float, ...], float] | None = None
        self.prompt_in_flight = False
        self.status = "Click for point or drag for box"

        image_topic = str(self.get_parameter("image_topic").value)
        preview_topic = str(self.get_parameter("preview_topic").value)
        mask_topic = str(self.get_parameter("mask_topic").value)
        result_topic = str(self.get_parameter("result_topic").value)
        if self.use_preview:
            self.create_subscription(
                Image, preview_topic, self.on_preview, qos_profile_sensor_data
            )
        else:
            self.create_subscription(
                Image, image_topic, self.on_image, qos_profile_sensor_data
            )
            self.create_subscription(
                Image, mask_topic, self.on_mask, qos_profile_sensor_data
            )
        self.create_subscription(String, result_topic, self.on_result, 10)
        self.add_client = self.create_client(AddObject, "/sam/add_object")
        self.reset_client = self.create_client(Trigger, "/sam/reset")
        display_fps = float(self.get_parameter("display_fps").value)
        self.display_period = 1.0 / display_fps

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        display_topic = preview_topic if self.use_preview else image_topic
        self.get_logger().info(
            f"interactive viewer on {display_topic}; "
            "click=point, drag=box, r=reset, q=quit"
        )

    @staticmethod
    def stamp_ns(message: Image) -> int:
        return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)

    def on_image(self, message: Image) -> None:
        frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        stamp = self.stamp_ns(message)
        self.frames[stamp] = frame
        self.frame_order.append(stamp)
        self.latest_frame = frame
        while len(self.frame_order) > 16:
            old_stamp = self.frame_order.popleft()
            self.frames.pop(old_stamp, None)
            self.masks.pop(old_stamp, None)
            self.results.pop(old_stamp, None)
        if self.latest_overlay is None:
            self.latest_overlay = frame.copy()

    def on_preview(self, message: Image) -> None:
        self.on_image(message)
        stamp = self.stamp_ns(message)
        frame = self.frames[stamp]
        self.latest_overlay = frame
        self.latest_overlay_stamp = stamp

    def on_mask(self, message: Image) -> None:
        match = _OBJECT_ID.search(message.header.frame_id)
        if match is None:
            return
        stamp = self.stamp_ns(message)
        mask = self.bridge.imgmsg_to_cv2(message, desired_encoding="mono8")
        self.masks.setdefault(stamp, {})[int(match.group(1))] = mask
        self.compose_overlay_if_ready(stamp)

    def on_result(self, message: String) -> None:
        try:
            result = json.loads(message.data)
        except json.JSONDecodeError:
            self.get_logger().warning("ignored invalid /sam/result_json payload")
            return
        stamp = int(result.get("stamp_ns", 0))
        frame = self.frames.get(stamp, self.latest_frame)
        if frame is None:
            return
        self.results[stamp] = result
        self.latest_result = result
        self.source_width = int(result.get("source_width", self.source_width))
        self.source_height = int(result.get("source_height", self.source_height))
        if self.use_preview:
            self.compose_ms = float(result.get("preview_compose_ms", 0.0))
        self.result_times.append(perf_counter())
        stamp_seconds = stamp * 1e-9
        if stamp > 0 and (
            not self.tracker_stamps or stamp_seconds > self.tracker_stamps[-1]
        ):
            self.tracker_stamps.append(stamp_seconds)
        if not self.use_preview:
            self.compose_overlay_if_ready(stamp)

    def compose_overlay_if_ready(self, stamp: int) -> None:
        frame = self.frames.get(stamp)
        result = self.results.get(stamp)
        if frame is None or result is None or stamp < self.latest_overlay_stamp:
            return
        expected = {int(object_id) for object_id in result.get("objects", [])}
        available = self.masks.get(stamp, {})
        if not all_masks_ready(expected, available):
            return
        compose_start = perf_counter()
        overlay = frame.copy()
        for object_id in expected:
            mask = available[object_id]
            color_index = (object_id - 1) % len(_COLORS)
            color = _COLORS[color_index]
            height, width = overlay.shape[:2]
            color_key = height, width, color_index
            color_layer = self.color_layers.get(color_key)
            if color_layer is None:
                color_layer = np.empty_like(overlay)
                color_layer[:] = color
                self.color_layers[color_key] = color_layer
            if self.blend_buffer is None or self.blend_buffer.shape != overlay.shape:
                self.blend_buffer = np.empty_like(overlay)
            x, y, region_width, region_height = cv2.boundingRect(mask)
            if region_width == 0 or region_height == 0:
                continue
            region = np.s_[
                y : y + region_height,
                x : x + region_width,
            ]
            cv2.addWeighted(
                overlay[region],
                0.55,
                color_layer[region],
                0.45,
                0.0,
                dst=self.blend_buffer[region],
            )
            cv2.copyTo(
                self.blend_buffer[region],
                mask[region],
                overlay[region],
            )
            if self.draw_contours:
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(
                    overlay, contours, -1, color, 2
                )
        self.latest_overlay = overlay
        self.latest_overlay_stamp = stamp
        self.compose_ms = (perf_counter() - compose_start) * 1000.0

    def on_mouse(self, event: int, x: int, y: int, flags: int, _: object) -> None:
        if self.latest_frame is None:
            return
        height, width = self.latest_frame.shape[:2]
        preview_point = display_to_image_point(
            x, y, self.current_scale, width, height
        )
        point = (
            (
                preview_point[0] * self.source_width / width,
                preview_point[1] * self.source_height / height,
            )
            if preview_point is not None
            else None
        )
        if event == cv2.EVENT_LBUTTONDOWN and point is not None:
            self.drag_start = point
            self.drag_current = point
            return
        if event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            if flags & cv2.EVENT_FLAG_LBUTTON and point is not None:
                self.drag_current = point
            return
        if event != cv2.EVENT_LBUTTONUP or self.drag_start is None:
            return
        start = self.drag_start
        self.drag_start = None
        self.drag_current = None
        if point is None:
            return
        box = drag_to_box(
            start,
            point,
            self.source_width,
            self.source_height,
            self.box_drag_min_pixels,
        )
        if box is None:
            self.send_prompt(AddObject.Request.POINT, point[0], point[1], 0.0, 0.0)
            self.prompt_marker = ("point", point, perf_counter())
        else:
            self.send_prompt(AddObject.Request.BOX, *box)
            self.prompt_marker = ("box", box, perf_counter())

    def send_prompt(self, kind: int, x0: float, y0: float, x1: float, y1: float) -> None:
        if self.prompt_in_flight:
            self.status = "Wait for the previous prompt"
            return
        if not self.add_client.service_is_ready():
            self.status = "Tracker service is not ready"
            return
        prompt = kind, x0, y0, x1, y1
        if self.replace_on_prompt:
            if not self.reset_client.service_is_ready():
                self.status = "Reset service is not ready"
                return
            self.prompt_in_flight = True
            self.status = "Resetting previous object"
            future = self.reset_client.call_async(Trigger.Request())
            future.add_done_callback(lambda done: self.after_prompt_reset(done, prompt))
            return
        self.submit_prompt(prompt)

    def after_prompt_reset(
        self,
        future: object,
        prompt: tuple[int, float, float, float, float],
    ) -> None:
        try:
            response = future.result()
            if not response.success:
                self.status = f"Reset failed: {response.message}"
                self.prompt_in_flight = False
                return
        except Exception as error:
            self.status = f"Reset failed: {error}"
            self.prompt_in_flight = False
            return
        self.masks.clear()
        self.results.clear()
        self.latest_result = {}
        self.latest_overlay_stamp = 0
        if self.latest_frame is not None:
            self.latest_overlay = self.latest_frame.copy()
        self.submit_prompt(prompt)

    def submit_prompt(self, prompt: tuple[int, float, float, float, float]) -> None:
        kind, x0, y0, x1, y1 = prompt
        self.prompt_in_flight = True
        request = AddObject.Request()
        request.kind = kind
        request.x0, request.y0 = float(x0), float(y0)
        request.x1, request.y1 = float(x1), float(y1)
        mode = "point" if kind == AddObject.Request.POINT else "box"
        self.status = f"Submitting {mode} prompt"
        future = self.add_client.call_async(request)
        future.add_done_callback(lambda done: self.on_prompt_response(done, mode))

    def on_prompt_response(self, future: object, mode: str) -> None:
        self.prompt_in_flight = False
        try:
            response = future.result()
            self.status = (
                f"Tracking object {response.object_id} ({mode})"
                if response.success
                else f"Prompt failed: {response.message}"
            )
        except Exception as error:
            self.status = f"Prompt failed: {error}"

    def reset(self) -> None:
        if not self.reset_client.service_is_ready():
            self.status = "Reset service is not ready"
            return
        self.reset_client.call_async(Trigger.Request())
        self.masks.clear()
        self.results.clear()
        self.latest_result = {}
        self.latest_overlay_stamp = 0
        if self.latest_frame is not None:
            self.latest_overlay = self.latest_frame.copy()
        self.prompt_marker = None
        self.status = "Reset; click for point or drag for box"

    def display(self) -> None:
        if self.latest_overlay is None:
            return
        display_start = perf_counter()
        if self.latest_overlay_stamp != self.last_presented_stamp:
            self.present_times.append(display_start)
            self.last_presented_stamp = self.latest_overlay_stamp
        frame = self.latest_overlay.copy()
        height, width = frame.shape[:2]
        scale = self.display_scale
        if self.display_max_width > 0:
            scale = min(scale, self.display_max_width / width)
        self.current_scale = scale
        if scale != 1.0:
            frame = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        self.draw_interaction(frame)
        self.draw_metrics(frame)
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        self.display_ms = (perf_counter() - display_start) * 1000.0
        now = perf_counter()
        if now - self.last_metrics_log >= 2.0:
            self.get_logger().info(
                "viewer_metrics "
                f"tracker_fps={event_rate_hz(self.tracker_stamps):.2f} "
                f"ui_receive_fps={event_rate_hz(self.result_times):.2f} "
                f"present_fps={event_rate_hz(self.present_times):.2f} "
                f"compose_ms={self.compose_ms:.2f} display_ms={self.display_ms:.2f}"
            )
            self.last_metrics_log = now
        if key in {27, ord("q")}:
            rclpy.shutdown()
        elif key == ord("r"):
            self.reset()

    def draw_interaction(self, frame: np.ndarray) -> None:
        scale_x = frame.shape[1] / self.source_width
        scale_y = frame.shape[0] / self.source_height

        def display_point(point: tuple[float, ...]) -> tuple[int, int]:
            return int(point[0] * scale_x), int(point[1] * scale_y)

        if self.drag_start is not None and self.drag_current is not None:
            start = display_point(self.drag_start)
            end = display_point(self.drag_current)
            cv2.rectangle(frame, start, end, (0, 255, 255), 2)
        marker = self.prompt_marker
        if marker is None or perf_counter() - marker[2] > 0.7:
            return
        if marker[0] == "point":
            cv2.circle(frame, display_point(marker[1]), 7, (0, 255, 255), -1)
        else:
            x0, y0, x1, y1 = marker[1]
            cv2.rectangle(
                frame,
                display_point((x0, y0)),
                display_point((x1, y1)),
                (0, 255, 255),
                2,
            )

    def draw_metrics(self, frame: np.ndarray) -> None:
        result = self.latest_result
        tracker_fps = event_rate_hz(self.tracker_stamps)
        receive_fps = event_rate_hz(self.result_times)
        present_fps = event_rate_hz(self.present_times)
        objects = len(result.get("objects", []))
        line1 = f"{self.status} | objects={objects}"
        line2 = (
            f"tracker={tracker_fps:.1f}  UI-rx={receive_fps:.1f}  "
            f"present={present_fps:.1f} FPS"
        )
        line3 = (
            f"infer={float(result.get('inference_ms', 0.0)):.1f} ms  "
            f"source-age={float(result.get('source_age_ms', 0.0)):.1f} ms  "
            f"compose={self.compose_ms:.1f} ms  display={self.display_ms:.1f} ms  "
            f"drops={int(result.get('dropped_frames', 0))}"
        )
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 92), (0, 0, 0), -1)
        cv2.putText(frame, line1, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, line2, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, line3, (12, 79), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(
            frame,
            "click: point | drag: box | r: reset | q: quit",
            (12, frame.shape[0] - 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


def main() -> None:
    rclpy.init()
    node = InteractiveViewer()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = Thread(
        target=executor.spin,
        name="sam2-trt-viewer-ros",
        daemon=True,
    )
    spin_thread.start()
    try:
        next_display = perf_counter()
        while rclpy.ok() and spin_thread.is_alive():
            node.display()
            next_display += node.display_period
            delay = next_display - perf_counter()
            if delay > 0:
                sleep(delay)
            else:
                next_display = perf_counter()
    finally:
        executor.shutdown()
        spin_thread.join(timeout=5.0)
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
