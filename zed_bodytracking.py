"""ZED 2i Body Tracking 关节角采集（多进程 GUI，输出格式与 RealSense 版一致）。"""

from __future__ import annotations

import multiprocessing as mp_ctx
import os
import time

# 采集子进程不初始化 TkAgg，避免与主进程 OpenCV/matplotlib GUI 争用 GIL
if mp_ctx.current_process().name != "MainProcess":
    os.environ.setdefault("MPLBACKEND", "Agg")
from pathlib import Path
from queue import Empty, Full
from typing import Optional

import cv2
import numpy as np
import pyzed.sl as sl

from datetime import datetime

from joint_angles import (
    FK_CONNECTIONS,
    ForwardKinematics,
    JointAngleCalculator,
    ReconstructedPoseVisualizer,
    figure_is_alive,
    pump_matplotlib_events,
)
from realsense_mediapipe import (
    BroadcastRecorder,
    OneEuroFilter3D,
    Pose3DVisualizer,
)
from zed_skeleton import (
    NUM_MP_KEYPOINTS,
    NUM_ZED_KEYPOINTS,
    draw_skeleton_2d,
    zed_body34_connections,
    zed_keypoints_to_unity,
    zed_viz_keypoints_to_unity,
)

LOCAL_DEBUG = True
FRAME_TIMEOUT_MS = 10000
WARMUP_GRAB_COUNT = 30
BODY_DETECTION_CONFIDENCE = 40
CV_WINDOW_NAME = "Motion Capture (ZED)"
RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"

ENABLE_ONE_EURO_FILTER = True
KEYPOINT_CONF_THRESHOLD = 0.0
FK_KNEE_SMOOTH_ALPHA = 0.15  # 越小越平滑，0.15 ≈ 时间常数约 6 帧@30fps


class FKAngleSmoother:
    """仅用于 FK 可视化：平滑膝部关节角，避免偶发帧抖动。

    录制数据不经过此平滑，保留原始角度。
    对 Knee_AX/AY/AZ 均做 EMA，以抑制 ZED 关键点位置噪声。
    """

    def __init__(self, alpha: float = FK_KNEE_SMOOTH_ALPHA) -> None:
        self.alpha = alpha
        self._prev: dict[str, float] = {}

    def for_fk(self, angles: dict[str, float]) -> dict[str, float]:
        if not angles:
            return angles
        out = dict(angles)
        for key in angles:
            if not key.startswith("Knee_"):
                continue
            val = angles[key]
            if key in self._prev:
                out[key] = self.alpha * val + (1.0 - self.alpha) * self._prev[key]
            self._prev[key] = out[key]
        return out


class ZEDBroadcastRecorder(BroadcastRecorder):
    """录制文件名使用 joint_angles_zed_ 前缀以区分设备。"""

    def start(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.output_dir / (
            f"joint_angles_zed_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        )
        import json

        self._file = self._path.open("w", encoding="utf-8")
        self._file.write(json.dumps({
            "type": "session_start",
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False) + "\n")
        self.recording = True
        self.recorded_frames = 0
        print(f"[录制] 已开始 → {self._path}")
        return self._path


class ZEDPose3DVisualizer(Pose3DVisualizer):
    """ZED BODY_34 全骨架 3D 可视化（viz_keypoints 已转为 MediaPipe 绘图 Y 约定）。"""

    def update(self, keypoints: list) -> None:
        super().update(keypoints)
        if self._closed or not figure_is_alive(self.fig):
            return
        valid_count = sum(1 for kp in keypoints if self._is_valid(kp))
        self.ax.set_title(f"3D Pose — ZED BODY_34 ({valid_count}/{NUM_ZED_KEYPOINTS})")
        self.fig.canvas.draw_idle()


class ZEDFKVisualizer(ReconstructedPoseVisualizer):
    """FK 重建窗口：+Y 头数据适配 matplotlib 竖直方向。"""

    @staticmethod
    def _unity_to_plot(p: np.ndarray) -> tuple[float, float, float]:
        return float(p[0]), float(p[2]), -float(p[1])

    def update(self, nodes: dict[str, np.ndarray]) -> None:
        if self._closed or not figure_is_alive(self.fig):
            return

        self.ax.cla()
        self.ax.set_xlabel("X (m, right)")
        self.ax.set_ylabel("Z (m, +Z → face)")
        self.ax.set_zlabel("Y (m, +Y → head)")
        self.ax.set_title(f"FK Reconstructed ({len(nodes)} nodes)")

        if not nodes:
            self.fig.canvas.draw_idle()
            return

        pts = list(nodes.values())
        plot_pts = [self._unity_to_plot(p) for p in pts]
        self.ax.scatter(
            [p[0] for p in plot_pts], [p[1] for p in plot_pts], [p[2] for p in plot_pts],
            c="mediumseagreen", s=40, depthshade=True,
        )

        for a, b in FK_CONNECTIONS:
            if a in nodes and b in nodes:
                p1 = self._unity_to_plot(nodes[a])
                p2 = self._unity_to_plot(nodes[b])
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                             color="royalblue", linewidth=2)

        xs = [p[0] for p in plot_pts]
        ys = [p[1] for p in plot_pts]
        zs = [p[2] for p in plot_pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        cz = sum(zs) / len(zs)
        span = max(
            max(abs(v - c) for v in coords)
            for coords, c in ((xs, cx), (ys, cy), (zs, cz))
        )
        half = max(0.5, span * 1.3)
        self.ax.set_xlim(cx - half, cx + half)
        self.ax.set_ylim(cy - half, cy + half)
        self.ax.set_zlim(cz - half, cz + half)
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.invert_zaxis()
        self.fig.canvas.draw_idle()


class ZEDCaptureWorker:
    """子进程：ZED 取流 + Body Tracking + 关节角计算（无 GUI）。"""

    def __init__(self) -> None:
        self.zed: Optional[sl.Camera] = None
        self.frame_count = 0
        self.start_time = time.time()
        self.camera_initialized = False
        self._frame_error_count = 0
        self._image_scale = 1.0
        self.one_euro_filters = [OneEuroFilter3D() for _ in range(NUM_MP_KEYPOINTS)]
        self.joint_calculator = JointAngleCalculator()
        self.forward_kinematics = ForwardKinematics()
        self._bodies = sl.Bodies()
        self._image = sl.Mat()
        self._body_runtime = sl.BodyTrackingRuntimeParameters()
        self._body_runtime.detection_confidence_threshold = BODY_DETECTION_CONFIDENCE

    def _apply_one_euro_filter(
        self, keypoints: list[dict], timestamp: float
    ) -> list[dict]:
        if not ENABLE_ONE_EURO_FILTER:
            return keypoints

        filtered_keypoints = []
        for kp in keypoints:
            idx = kp["id"]
            point = np.array([kp["x"], kp["y"], kp["z"]], dtype=np.float64)
            update = kp["confidence"] > KEYPOINT_CONF_THRESHOLD
            smoothed = self.one_euro_filters[idx].filter(point, timestamp, update)
            filtered_keypoints.append({
                **kp,
                "x": float(smoothed[0]),
                "y": float(smoothed[1]),
                "z": float(smoothed[2]),
            })
        return filtered_keypoints

    @staticmethod
    def _select_body(body_list) -> Optional[object]:
        if not body_list:
            return None
        best = body_list[0]
        best_conf = getattr(best, "confidence", 0.0) or 0.0
        for body in body_list[1:]:
            conf = getattr(body, "confidence", 0.0) or 0.0
            if conf > best_conf:
                best = body
                best_conf = conf
        return best

    def _warmup_camera(self) -> None:
        print(f"正在预热 ZED（丢弃前 {WARMUP_GRAB_COUNT} 帧）...")
        for i in range(WARMUP_GRAB_COUNT):
            if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"预热第 {i + 1}/{WARMUP_GRAB_COUNT} 帧 grab 失败")
        print("ZED 预热完成")

    def initialize(self) -> None:
        print("[采集进程] 正在初始化 ZED 相机...")
        devices = sl.Camera().get_device_list()
        if not devices:
            raise RuntimeError(
                "未检测到 ZED 设备。请确认相机已连接并使用 USB 3.0 接口。"
            )
        print(f"检测到 {len(devices)} 台设备，正在打开 SN {devices[0].serial_number} ...")

        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.camera_fps = 30
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL_LIGHT
        init_params.coordinate_units = sl.UNIT.METER
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.open_timeout_sec = 15.0

        last_status = sl.ERROR_CODE.CAMERA_NOT_DETECTED
        for attempt in range(3):
            if attempt > 0:
                print(f"打开失败，{2.0:.0f}s 后重试 ({attempt + 1}/3)...")
                time.sleep(2.0)
                if self.zed is not None:
                    try:
                        self.zed.close()
                    except Exception:
                        pass
                self.zed = sl.Camera()

            if self.zed is None:
                self.zed = sl.Camera()
            status = self.zed.open(init_params)
            if status == sl.ERROR_CODE.SUCCESS:
                break
            last_status = status
        else:
            raise RuntimeError(
                f"无法打开 ZED 相机: {last_status}\n"
                "请确认:\n"
                "1. 已关闭 ZED Explorer 及其他占用相机的程序\n"
                "2. 无其他 zed_bodytracking.py / python 实例在运行（任务管理器结束 python.exe）\n"
                "3. 重新插拔 USB 后等待 5 秒再试"
            )

        cam_info = self.zed.get_camera_information()
        serial = cam_info.serial_number if cam_info else "unknown"
        model = cam_info.camera_model if cam_info else "ZED"
        print(f"检测到设备: {model} (SN: {serial})")

        tracking_params = sl.PositionalTrackingParameters()
        tracking_params.set_floor_as_origin = True
        status = self.zed.enable_positional_tracking(tracking_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Positional Tracking 启用失败: {status}")

        body_params = sl.BodyTrackingParameters()
        body_params.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST
        body_params.body_format = sl.BODY_FORMAT.BODY_34
        body_params.enable_tracking = True
        body_params.enable_body_fitting = True
        status = self.zed.enable_body_tracking(body_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Body Tracking 启用失败: {status}")

        self._warmup_camera()

        self.camera_initialized = True
        print("[采集进程] ZED 初始化成功！")

    def shutdown(self) -> None:
        if self.zed is not None:
            if self.camera_initialized:
                try:
                    self.zed.disable_body_tracking()
                except Exception:
                    pass
                try:
                    self.zed.disable_positional_tracking()
                except Exception:
                    pass
            try:
                self.zed.close()
            except Exception:
                pass
            self.zed = None
        self.camera_initialized = False
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0.0
        print(f"[采集进程] 已停止: {self.frame_count} 帧 | 平均 FPS: {avg_fps:.1f}")

    def run(self, frame_queue: mp_ctx.Queue, stop_event: mp_ctx.Event) -> None:
        try:
            self.initialize()
            frame_queue.put({"type": "ready"})
            print("[采集进程] 开始捕获...")

            while not stop_event.is_set():
                packet = self.process_frame()
                if packet is None:
                    continue
                try:
                    frame_queue.put(packet, block=False)
                except Full:
                    try:
                        frame_queue.get_nowait()
                    except Empty:
                        pass
                    try:
                        frame_queue.put(packet, block=False)
                    except Full:
                        pass
        except Exception as e:
            frame_queue.put({"type": "error", "message": str(e)})
            print(f"[采集进程] 运行出错: {e}")
        finally:
            self.shutdown()

    def process_frame(self) -> Optional[dict]:
        if not self.camera_initialized or self.zed is None:
            return None

        frame_start = time.time()
        self.frame_count += 1

        try:
            if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
                return None

            self.zed.retrieve_image(self._image, sl.VIEW.LEFT)
            self.zed.retrieve_bodies(self._bodies, self._body_runtime)

            color_image = self._image.get_data()
            if color_image is None:
                return None
            color_image = np.ascontiguousarray(color_image.copy())
            self._image_scale = self._image.get_width() / 1280.0

            timestamp = time.time()
            joint_angles: dict = {}
            keypoints: list[dict] = []
            viz_keypoints: list[dict] = []

            body = self._select_body(self._bodies.body_list)
            body_count = len(self._bodies.body_list)
            if body is not None:
                viz_keypoints = zed_viz_keypoints_to_unity(body)
                raw_keypoints = zed_keypoints_to_unity(body)
                if raw_keypoints:
                    keypoints = self._apply_one_euro_filter(raw_keypoints, timestamp)
                    self.forward_kinematics.update_lengths(keypoints)
                    joint_angles = self.joint_calculator.compute(keypoints)
                draw_skeleton_2d(color_image, body, self._image_scale)

            elapsed = time.time() - frame_start
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            valid_count = sum(1 for kp in viz_keypoints if kp.get("source") == "zed")
            cv2.putText(color_image, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(color_image, f"Bodies: {body_count}  Nodes: {valid_count}/{NUM_ZED_KEYPOINTS}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(color_image, "Joint Angles (ZED)", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

            self._frame_error_count = 0
            return {
                "type": "frame",
                "timestamp": timestamp,
                "frame_count": self.frame_count,
                "fps": fps,
                "joint_angles": joint_angles,
                "keypoints": keypoints,
                "viz_keypoints": viz_keypoints,
                "color_image": color_image,
            }

        except Exception as e:
            self._frame_error_count += 1
            if self._frame_error_count == 1 or self._frame_error_count % 10 == 0:
                print(f"[采集进程] 处理帧时出错: {e}")
            return None


class ZEDMotionCaptureApp:
    """主进程：OpenCV + Matplotlib 可视化（与 ZED 采集进程隔离）。"""

    def __init__(
        self,
        frame_queue: mp_ctx.Queue,
        stop_event: mp_ctx.Event,
        capture_process: mp_ctx.Process,
    ) -> None:
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.capture_process = capture_process
        self.pose_connections = zed_body34_connections()
        self.visualizer: Optional[ZEDPose3DVisualizer] = None
        self.fk_visualizer: Optional[ReconstructedPoseVisualizer] = None
        self.forward_kinematics = ForwardKinematics()
        self.start_time = time.time()
        self.frame_count = 0
        self.recorder = ZEDBroadcastRecorder(RECORDINGS_DIR)
        self._fk_angle_smoother = FKAngleSmoother()
        self._record_btn_rect: tuple[int, int, int, int] = (0, 0, 0, 0)

    def _init_visualizers(self) -> None:
        if LOCAL_DEBUG:
            self.visualizer = ZEDPose3DVisualizer(self.pose_connections)
            self.fk_visualizer = ZEDFKVisualizer()
            cv2.namedWindow(CV_WINDOW_NAME)
            cv2.setMouseCallback(CV_WINDOW_NAME, self._on_mouse)
            print("[主进程] 3D 可视化窗口已创建（ZED Body + 关节角 FK 对比）")
            print("[主进程] 点击 REC 按钮或按 R 键开始/停止录制")

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x1, y1, x2, y2 = self._record_btn_rect
        if x1 <= x <= x2 and y1 <= y <= y2:
            self.recorder.toggle()

    def _draw_record_button(self, image: np.ndarray) -> None:
        h, w = image.shape[:2]
        bw, bh = 150, 40
        x1, y1 = 10, h - bh - 10
        x2, y2 = x1 + bw, y1 + bh
        self._record_btn_rect = (x1, y1, x2, y2)

        bg = (0, 0, 200) if self.recorder.recording else (50, 50, 50)
        cv2.rectangle(image, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 2)
        label = "STOP REC" if self.recorder.recording else "REC"
        cv2.putText(image, label, (x1 + 12, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        if self.recorder.recording:
            cv2.circle(image, (x2 - 18, y1 + bh // 2), 7, (0, 0, 255), -1)
            cv2.putText(image, str(self.recorder.recorded_frames),
                        (x2 + 8, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def _handle_frame(self, packet: dict) -> None:
        self.frame_count += 1
        keypoints = packet.get("keypoints", [])
        viz_keypoints = packet.get("viz_keypoints", keypoints)
        joint_angles = packet.get("joint_angles", {})
        color_image = packet["color_image"].copy()

        self.recorder.write(packet)
        self._draw_record_button(color_image)

        cv2.imshow(CV_WINDOW_NAME, color_image)

        if self.visualizer and viz_keypoints:
            self.visualizer.update(viz_keypoints)

        if self.fk_visualizer and joint_angles:
            self.forward_kinematics.update_lengths(keypoints)
            fk_angles = self._fk_angle_smoother.for_fk(joint_angles)
            fk_nodes = self.forward_kinematics.rebuild(fk_angles)
            self.fk_visualizer.update(fk_nodes)

        pump_matplotlib_events()

    def _drain_queue(self) -> tuple[Optional[dict], bool]:
        latest: Optional[dict] = None
        should_stop = False
        while True:
            try:
                msg = self.frame_queue.get_nowait()
            except Empty:
                break
            msg_type = msg.get("type")
            if msg_type == "frame":
                latest = msg
            elif msg_type == "error":
                print(f"[主进程] 采集进程报错: {msg.get('message')}")
                should_stop = True
            elif msg_type == "ready":
                print("[主进程] 采集进程就绪")
        return latest, should_stop

    def run(self) -> None:
        self._init_visualizers()
        print("[主进程] 等待采集进程就绪...")

        try:
            while not self.stop_event.is_set():
                if not self.capture_process.is_alive():
                    self._drain_queue()
                    print("[主进程] 采集进程已退出")
                    break

                latest, should_stop = self._drain_queue()
                if should_stop:
                    break
                if latest is not None:
                    self._handle_frame(latest)

                pump_matplotlib_events()
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[主进程] 用户请求退出")
                    break
                if key == ord("r"):
                    self.recorder.toggle()

                time.sleep(0.005)

        except KeyboardInterrupt:
            print("[主进程] 正在关闭...")
        finally:
            self.stop()

    def stop(self) -> None:
        self.stop_event.set()
        self.recorder.stop()
        if self.capture_process.is_alive():
            self.capture_process.join(timeout=8.0)
            if self.capture_process.is_alive():
                print("[主进程] 采集进程未响应，强制终止")
                self.capture_process.terminate()
                self.capture_process.join(timeout=2.0)

        if self.visualizer:
            self.visualizer.close()
        if self.fk_visualizer:
            self.fk_visualizer.close()
        cv2.destroyAllWindows()

        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        print(f"[主进程] 已停止: 显示 {self.frame_count} 帧 | 平均 FPS: {avg_fps:.1f}")


def capture_process_entry(
    frame_queue: mp_ctx.Queue,
    stop_event: mp_ctx.Event,
) -> None:
    worker = ZEDCaptureWorker()
    worker.run(frame_queue, stop_event)


def main() -> None:
    ctx = mp_ctx.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=2)
    stop_event = ctx.Event()

    capture_proc = ctx.Process(
        target=capture_process_entry,
        args=(frame_queue, stop_event),
        name="ZEDCapture",
        daemon=False,
    )
    capture_proc.start()

    app = ZEDMotionCaptureApp(frame_queue, stop_event, capture_proc)
    app.run()


if __name__ == "__main__":
    mp_ctx.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序异常退出: {e}")
