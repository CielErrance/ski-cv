"""ZED 相机 + MediaPipe 姿态检测关节角采集（多进程 GUI）。

使用 ZED 左目 RGB 运行 MediaPipe PoseLandmarker，并用 ZED 深度/点云
校正 world landmarks 的米制尺度；输出格式与 realsense_mediapipe.py 一致。

运行（请先 conda activate ski）:
    python zed_mediapipe.py
"""

from __future__ import annotations

import multiprocessing as mp_ctx
import os
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Full
from typing import Optional

import cv2
import mediapipe
import numpy as np
import pyzed.sl as sl
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
    drawing_utils,
)
from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmark,
    PoseLandmarksConnections,
)

if mp_ctx.current_process().name == "MainProcess":
    import matplotlib
    matplotlib.use("TkAgg", force=True)
else:
    os.environ.setdefault("MPLBACKEND", "Agg")

from joint_angles import (
    CombinedPose3DVisualizer,
    ForwardKinematics,
    JointAngleCalculator,
)
from realsense_mediapipe import (
    BroadcastRecorder,
    OneEuroFilter3D,
    ensure_pose_model,
)

LOCAL_DEBUG = True
WARMUP_GRAB_COUNT = 30
DEPTH_SAMPLE_RADIUS = 2
MIN_VALID_DEPTH_M = 0.2
MAX_VALID_DEPTH_M = 5.0
NUM_KEYPOINTS = 33

ENABLE_ONE_EURO_FILTER = True
KEYPOINT_CONF_THRESHOLD = 0.0

CV_WINDOW_NAME = "Motion Capture (ZED + MediaPipe)"
CV_DISPLAY_WIDTH = 960
RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"


class ZEDMPBroadcastRecorder(BroadcastRecorder):
    """录制文件名使用 joint_angles_zed_mp_ 前缀。"""

    def start(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.output_dir / (
            f"joint_angles_zed_mp_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
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


class ZEDMediaPipeCaptureWorker:
    """子进程：ZED 取流 + MediaPipe 推理 + 关节角计算（无 GUI）。"""

    def __init__(self) -> None:
        self.zed: Optional[sl.Camera] = None
        self.pose_landmarker: Optional[PoseLandmarker] = None
        self.pose_connections = PoseLandmarksConnections.POSE_LANDMARKS
        self.frame_count = 0
        self.frame_timestamp_ms = 0
        self.start_time = time.time()
        self.camera_initialized = False
        self._frame_error_count = 0
        self._last_world_scale = 1.0
        self._image_width = 1280
        self._image_height = 720
        self.one_euro_filters = [OneEuroFilter3D() for _ in range(NUM_KEYPOINTS)]
        self.joint_calculator = JointAngleCalculator()
        self.forward_kinematics = ForwardKinematics()
        self._image = sl.Mat()
        self._xyz = sl.Mat()

    @staticmethod
    def _landmark_pixel(landmark, width: int, height: int) -> tuple[int, int]:
        return int(landmark.x * width), int(landmark.y * height)

    @staticmethod
    def _world_vec(landmark) -> np.ndarray:
        return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)

    @staticmethod
    def _world_to_unity(relative_world: np.ndarray, scale: float) -> np.ndarray:
        """MediaPipe world → Unity（髋原点，+Y 头，+Z 脸）。"""
        return np.array([
            relative_world[0] * scale,
            -relative_world[1] * scale,
            relative_world[2] * scale,
        ], dtype=np.float64)

    def _sample_xyz_m(self, x_px: int, y_px: int) -> Optional[np.ndarray]:
        """从左目对齐的 XYZ 点云读取相机坐标系下的 3D 点（米）。"""
        if not (0 <= x_px < self._image_width and 0 <= y_px < self._image_height):
            return None

        samples: list[np.ndarray] = []
        for dy in range(-DEPTH_SAMPLE_RADIUS, DEPTH_SAMPLE_RADIUS + 1):
            for dx in range(-DEPTH_SAMPLE_RADIUS, DEPTH_SAMPLE_RADIUS + 1):
                px, py = x_px + dx, y_px + dy
                if not (0 <= px < self._image_width and 0 <= py < self._image_height):
                    continue
                err, value = self._xyz.get_value(px, py)
                if err != sl.ERROR_CODE.SUCCESS:
                    continue
                point = np.array([float(value[0]), float(value[1]), float(value[2])])
                if not np.all(np.isfinite(point)):
                    continue
                depth_forward = point[2]
                if MIN_VALID_DEPTH_M < depth_forward < MAX_VALID_DEPTH_M:
                    samples.append(point)

        if not samples:
            return None
        return np.median(samples, axis=0)

    def _collect_depth_points_for_scale(
        self,
        landmarks_2d,
    ) -> dict[int, np.ndarray]:
        depth_points: dict[int, np.ndarray] = {}
        for idx, landmark in enumerate(landmarks_2d):
            x_px, y_px = self._landmark_pixel(
                landmark, self._image_width, self._image_height,
            )
            point = self._sample_xyz_m(x_px, y_px)
            if point is not None:
                depth_points[idx] = point
        return depth_points

    def _estimate_world_scale(
        self,
        world_landmarks,
        depth_points: dict[int, np.ndarray],
    ) -> float:
        scale_candidates: list[float] = []

        def add_pair_scale(idx_a: int, idx_b: int) -> None:
            if idx_a not in depth_points or idx_b not in depth_points:
                return
            real_dist = float(np.linalg.norm(depth_points[idx_a] - depth_points[idx_b]))
            world_dist = float(np.linalg.norm(
                self._world_vec(world_landmarks[idx_a])
                - self._world_vec(world_landmarks[idx_b])
            ))
            if world_dist > 1e-4 and real_dist > 1e-4:
                scale_candidates.append(real_dist / world_dist)

        add_pair_scale(PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER)
        add_pair_scale(PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP)

        if scale_candidates:
            return float(np.median(scale_candidates))
        return self._last_world_scale

    def _make_keypoint(
        self,
        idx: int,
        point: np.ndarray,
        confidence: float,
        source: str,
    ) -> dict:
        return {
            "id": idx,
            "name": PoseLandmark(idx).name,
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]),
            "confidence": confidence,
            "source": source,
        }

    def _build_world_keypoints(
        self,
        landmarks_2d,
        world_landmarks,
    ) -> list[dict]:
        depth_points = self._collect_depth_points_for_scale(landmarks_2d)
        scale = self._estimate_world_scale(world_landmarks, depth_points)
        self._last_world_scale = scale
        hip_world = 0.5 * (
            self._world_vec(world_landmarks[PoseLandmark.LEFT_HIP])
            + self._world_vec(world_landmarks[PoseLandmark.RIGHT_HIP])
        )

        keypoints: list[dict] = []
        for idx, landmark in enumerate(landmarks_2d):
            relative_world = self._world_vec(world_landmarks[idx]) - hip_world
            point = self._world_to_unity(relative_world, scale)
            confidence = landmark.visibility if landmark.visibility is not None else 0.0
            keypoints.append(self._make_keypoint(idx, point, confidence, "zed_mp"))
        return keypoints

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

    def _warmup_camera(self) -> None:
        print(f"正在预热 ZED（丢弃前 {WARMUP_GRAB_COUNT} 帧）...")
        for i in range(WARMUP_GRAB_COUNT):
            if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"预热第 {i + 1}/{WARMUP_GRAB_COUNT} 帧 grab 失败")
        print("ZED 预热完成")

    def _init_pose_landmarker(self) -> None:
        print("正在加载 MediaPipe 姿态检测模型...")
        model_path = str(ensure_pose_model())
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.pose_landmarker = PoseLandmarker.create_from_options(options)
        print("MediaPipe 模型加载完成")

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
                print(f"打开失败，2s 后重试 ({attempt + 1}/3)...")
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
                "请确认已关闭 ZED Explorer 及其他占用相机的程序。"
            )

        cam_info = self.zed.get_camera_information()
        res = cam_info.camera_configuration.resolution
        self._image_width = res.width
        self._image_height = res.height
        model = cam_info.camera_model if cam_info else "ZED"
        serial = cam_info.serial_number if cam_info else "unknown"
        print(f"检测到设备: {model} (SN: {serial}, {self._image_width}x{self._image_height})")

        self._warmup_camera()
        self._init_pose_landmarker()
        self.camera_initialized = True
        print("[采集进程] ZED + MediaPipe 初始化成功！")

    def shutdown(self) -> None:
        if self.pose_landmarker is not None:
            self.pose_landmarker.close()
            self.pose_landmarker = None
        if self.zed is not None:
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
        if not self.camera_initialized or self.zed is None or self.pose_landmarker is None:
            return None

        frame_start = time.time()
        self.frame_count += 1

        try:
            if self.zed.grab() != sl.ERROR_CODE.SUCCESS:
                return None

            self.zed.retrieve_image(self._image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self._xyz, sl.MEASURE.XYZ)

            color_image = self._image.get_data()
            if color_image is None:
                return None
            color_image = np.ascontiguousarray(color_image.copy())

            if color_image.shape[2] == 4:
                bgr_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
            elif color_image.shape[2] == 3:
                bgr_image = color_image
            else:
                raise ValueError(f"不支持的 ZED 图像通道数: {color_image.shape[2]}")
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            mp_image = mediapipe.Image(
                image_format=mediapipe.ImageFormat.SRGB, data=rgb_image,
            )
            self.frame_timestamp_ms += 33
            results = self.pose_landmarker.detect_for_video(
                mp_image, self.frame_timestamp_ms,
            )

            timestamp = time.time()
            joint_angles: dict = {}
            keypoints: list[dict] = []

            if results.pose_landmarks and results.pose_world_landmarks:
                raw_keypoints = self._build_world_keypoints(
                    results.pose_landmarks[0],
                    results.pose_world_landmarks[0],
                )
                keypoints = self._apply_one_euro_filter(raw_keypoints, timestamp)
                self.forward_kinematics.update_lengths(keypoints)
                joint_angles = self.joint_calculator.compute(keypoints)

                drawing_utils.draw_landmarks(
                    bgr_image,
                    results.pose_landmarks[0],
                    self.pose_connections,
                    drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2),
                    drawing_utils.DrawingSpec(
                        color=(255, 0, 0), thickness=2, circle_radius=2),
                )

            elapsed = time.time() - frame_start
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            valid_count = sum(1 for kp in keypoints if kp.get("source") == "zed_mp")
            scale_txt = f" scale={self._last_world_scale:.2f}" if keypoints else ""
            cv2.putText(bgr_image, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(
                bgr_image,
                f"MP Nodes: {valid_count}/33{scale_txt}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2,
            )
            cv2.putText(bgr_image, "ZED + MediaPipe", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

            self._frame_error_count = 0
            return {
                "type": "frame",
                "timestamp": timestamp,
                "frame_count": self.frame_count,
                "fps": fps,
                "joint_angles": joint_angles,
                "keypoints": keypoints,
                "color_image": bgr_image,
            }

        except Exception as e:
            self._frame_error_count += 1
            if self._frame_error_count == 1 or self._frame_error_count % 10 == 0:
                print(f"[采集进程] 处理帧时出错: {e}")
            return None


class ZEDMediaPipeApp:
    """主进程：OpenCV + 3D 可视化（与采集进程隔离）。"""

    def __init__(
        self,
        frame_queue: mp_ctx.Queue,
        stop_event: mp_ctx.Event,
        capture_process: mp_ctx.Process,
    ) -> None:
        self.frame_queue = frame_queue
        self.stop_event = stop_event
        self.capture_process = capture_process
        self.pose_connections = PoseLandmarksConnections.POSE_LANDMARKS
        self.visualizer: Optional[CombinedPose3DVisualizer] = None
        self.forward_kinematics = ForwardKinematics()
        self.start_time = time.time()
        self.frame_count = 0
        self.recorder = ZEDMPBroadcastRecorder(RECORDINGS_DIR)
        self._record_btn_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._display_scale = 1.0

    def _prepare_display(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if w <= CV_DISPLAY_WIDTH:
            self._display_scale = 1.0
            return image
        self._display_scale = CV_DISPLAY_WIDTH / w
        return cv2.resize(
            image,
            (CV_DISPLAY_WIDTH, int(h * self._display_scale)),
            interpolation=cv2.INTER_AREA,
        )

    def _init_visualizers(self) -> None:
        if LOCAL_DEBUG:
            self.visualizer = CombinedPose3DVisualizer(
                self.pose_connections,
                raw_kp_valid=lambda kp: kp.get("source") == "zed_mp",
                raw_title="MediaPipe (ZED depth)",
                fk_title="Joint Angles FK",
                window_title="Motion Capture 3D (ZED + MediaPipe)",
            )
            cv2.namedWindow(CV_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(CV_WINDOW_NAME, self._on_mouse)
            print("[主进程] 3D 对比窗口已创建（左：MediaPipe | 右：FK 重建）")
            print("[主进程] 3D 窗口：左键拖拽旋转 | 滚轮缩放 | 双击重置视角")
            print("[主进程] 点击 REC 按钮或按 R 键开始/停止录制")

    def _on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if self._display_scale > 0:
            x = int(x / self._display_scale)
            y = int(y / self._display_scale)
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
        joint_angles = packet.get("joint_angles", {})
        color_image = packet["color_image"].copy()

        self.recorder.write(packet)
        self._draw_record_button(color_image)
        cv2.imshow(CV_WINDOW_NAME, self._prepare_display(color_image))

        if self.visualizer:
            fk_nodes = None
            if joint_angles and keypoints:
                self.forward_kinematics.update_lengths(keypoints)
                fk_nodes = self.forward_kinematics.rebuild(joint_angles)
            valid_count = sum(
                1 for kp in keypoints if kp.get("source") == "zed_mp"
            ) if keypoints else 0
            self.visualizer.update(
                keypoints or None,
                fk_nodes,
                raw_title_suffix=f"({valid_count}/33)",
            )

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
        cv2.destroyAllWindows()

        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0.0
        print(f"[主进程] 已停止: 显示 {self.frame_count} 帧 | 平均 FPS: {avg_fps:.1f}")


def capture_process_entry(
    frame_queue: mp_ctx.Queue,
    stop_event: mp_ctx.Event,
) -> None:
    worker = ZEDMediaPipeCaptureWorker()
    worker.run(frame_queue, stop_event)


def main() -> None:
    ctx = mp_ctx.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=2)
    stop_event = ctx.Event()

    capture_proc = ctx.Process(
        target=capture_process_entry,
        args=(frame_queue, stop_event),
        name="ZEDMediaPipeCapture",
        daemon=False,
    )
    capture_proc.start()

    app = ZEDMediaPipeApp(frame_queue, stop_event, capture_proc)
    app.run()


if __name__ == "__main__":
    mp_ctx.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序异常退出: {e}")
