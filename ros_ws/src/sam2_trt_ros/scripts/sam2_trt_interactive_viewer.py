#!/usr/bin/env python3

from __future__ import annotations

import json
import re
from collections import deque
from time import perf_counter

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from sam2_trt.interactive import display_to_image_point, drag_to_box, event_rate_hz
from sam2_trt_msgs.srv import AddObject


_OBJECT_ID = re.compile(r"/sam_object_(\d+)$")
_COLORS = ((0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 0, 255))


class InteractiveViewer(Node):
    def __init__(self) -> None:
        super().__init__("sam2_trt_interactive_viewer")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("mask_topic", "/sam/object_masks")
        self.declare_parameter("result_topic", "/sam/result_json")
        self.declare_parameter("window_name", "SAM2 TensorRT tracking")
        self.declare_parameter("display_fps", 30.0)
        self.declare_parameter("display_scale", 1.0)
        self.declare_parameter("display_max_width", 1280)
        self.declare_parameter("box_drag_min_pixels", 5.0)

        self.bridge = CvBridge()
        self.window_name = str(self.get_parameter("window_name").value)
        self.display_scale = float(self.get_parameter("display_scale").value)
        self.display_max_width = int(self.get_parameter("display_max_width").value)
        self.box_drag_min_pixels = float(self.get_parameter("box_drag_min_pixels").value)
        self.current_scale = 1.0
        self.frames: dict[int, np.ndarray] = {}
        self.frame_order: deque[int] = deque()
        self.masks: dict[int, dict[int, np.ndarray]] = {}
        self.latest_frame: np.ndarray | None = None
        self.latest_overlay: np.ndarray | None = None
        self.latest_overlay_stamp = 0
        self.latest_result: dict[str, object] = {}
        self.result_times: deque[float] = deque(maxlen=120)
        self.drag_start: tuple[float, float] | None = None
        self.drag_current: tuple[float, float] | None = None
        self.prompt_marker: tuple[str, tuple[float, ...], float] | None = None
        self.status = "Click for point or drag for box"

        image_topic = str(self.get_parameter("image_topic").value)
        mask_topic = str(self.get_parameter("mask_topic").value)
        result_topic = str(self.get_parameter("result_topic").value)
        self.create_subscription(Image, image_topic, self.on_image, qos_profile_sensor_data)
        self.create_subscription(Image, mask_topic, self.on_mask, qos_profile_sensor_data)
        self.create_subscription(String, result_topic, self.on_result, 10)
        self.add_client = self.create_client(AddObject, "/sam/add_object")
        self.reset_client = self.create_client(Trigger, "/sam/reset")
        display_fps = float(self.get_parameter("display_fps").value)
        self.create_timer(1.0 / display_fps, self.display)

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        self.get_logger().info(
            f"interactive viewer on {image_topic}; click=point, drag=box, r=reset, q=quit"
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
        if self.latest_overlay is None:
            self.latest_overlay = frame.copy()

    def on_mask(self, message: Image) -> None:
        match = _OBJECT_ID.search(message.header.frame_id)
        if match is None:
            return
        stamp = self.stamp_ns(message)
        mask = self.bridge.imgmsg_to_cv2(message, desired_encoding="mono8")
        self.masks.setdefault(stamp, {})[int(match.group(1))] = mask
        self.compose_overlay(stamp)

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
        self.latest_result = result
        self.result_times.append(perf_counter())
        self.compose_overlay(stamp)

    def compose_overlay(self, stamp: int) -> None:
        frame = self.frames.get(stamp)
        if frame is None or stamp < self.latest_overlay_stamp:
            return
        overlay = frame.copy()
        for object_id, mask in self.masks.get(stamp, {}).items():
            selected = mask > 0
            color = np.asarray(_COLORS[(object_id - 1) % len(_COLORS)], dtype=np.float32)
            overlay[selected] = (
                overlay[selected].astype(np.float32) * 0.55 + color * 0.45
            ).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, tuple(int(value) for value in color), 2)
        self.latest_overlay = overlay
        self.latest_overlay_stamp = stamp

    def on_mouse(self, event: int, x: int, y: int, flags: int, _: object) -> None:
        if self.latest_frame is None:
            return
        height, width = self.latest_frame.shape[:2]
        point = display_to_image_point(x, y, self.current_scale, width, height)
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
        box = drag_to_box(start, point, width, height, self.box_drag_min_pixels)
        if box is None:
            self.send_prompt(AddObject.Request.POINT, point[0], point[1], 0.0, 0.0)
            self.prompt_marker = ("point", point, perf_counter())
        else:
            self.send_prompt(AddObject.Request.BOX, *box)
            self.prompt_marker = ("box", box, perf_counter())

    def send_prompt(self, kind: int, x0: float, y0: float, x1: float, y1: float) -> None:
        if not self.add_client.service_is_ready():
            self.status = "Tracker service is not ready"
            return
        request = AddObject.Request()
        request.kind = kind
        request.x0, request.y0 = float(x0), float(y0)
        request.x1, request.y1 = float(x1), float(y1)
        mode = "point" if kind == AddObject.Request.POINT else "box"
        self.status = f"Submitting {mode} prompt"
        future = self.add_client.call_async(request)
        future.add_done_callback(lambda done: self.on_prompt_response(done, mode))

    def on_prompt_response(self, future: object, mode: str) -> None:
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
        self.latest_result = {}
        self.prompt_marker = None
        self.status = "Reset; click for point or drag for box"

    def display(self) -> None:
        if self.latest_overlay is None:
            return
        frame = self.latest_overlay.copy()
        self.draw_interaction(frame)
        self.draw_metrics(frame)
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
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in {27, ord("q")}:
            rclpy.shutdown()
        elif key == ord("r"):
            self.reset()

    def draw_interaction(self, frame: np.ndarray) -> None:
        if self.drag_start is not None and self.drag_current is not None:
            start = tuple(int(value) for value in self.drag_start)
            end = tuple(int(value) for value in self.drag_current)
            cv2.rectangle(frame, start, end, (0, 255, 255), 2)
        marker = self.prompt_marker
        if marker is None or perf_counter() - marker[2] > 0.7:
            return
        if marker[0] == "point":
            cv2.circle(frame, (int(marker[1][0]), int(marker[1][1])), 7, (0, 255, 255), -1)
        else:
            x0, y0, x1, y1 = marker[1]
            cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 255), 2)

    def draw_metrics(self, frame: np.ndarray) -> None:
        result = self.latest_result
        output_fps = event_rate_hz(self.result_times)
        objects = len(result.get("objects", []))
        line1 = f"{self.status} | objects={objects} | output={output_fps:.1f} FPS"
        line2 = (
            f"infer={float(result.get('inference_ms', 0.0)):.1f} ms  "
            f"worker={float(result.get('worker_total_ms', 0.0)):.1f} ms  "
            f"source-age={float(result.get('source_age_ms', 0.0)):.1f} ms  "
            f"drops={int(result.get('dropped_frames', 0))}"
        )
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 66), (0, 0, 0), -1)
        cv2.putText(frame, line1, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, line2, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
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
    try:
        rclpy.spin(node)
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
