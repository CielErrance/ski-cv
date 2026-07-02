import multiprocessing as mp_ctx
import os

# 主进程固定 TkAgg；子进程用 Agg，避免与 OpenCV GUI 争用 GIL
if mp_ctx.current_process().name == "MainProcess":
    import matplotlib
    matplotlib.use("TkAgg", force=True)
else:
    os.environ.setdefault("MPLBACKEND", "Agg")

import cv2
import numpy as np
import pyrealsense2 as rs
import mediapipe
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
# import json
# import struct
import json
import time
import math
from queue import Empty, Full
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional, TextIO
from urllib.request import urlretrieve

from joint_angles import (
    CombinedPose3DVisualizer,
    ForwardKinematics,
    JointAngleCalculator,
    _plt,
    _set_axonometric_view,
    figure_is_alive,
)

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
POSE_MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_full.task"
POSE_MODEL_CACHE = (
    Path(os.environ.get("LOCALAPPDATA", Path.home())) / "cv_pose_models" / "pose_landmarker_full.task"
)


def ensure_pose_model() -> Path:
    """确保姿态检测模型存在。MediaPipe C 库无法读取中文路径，因此缓存到 ASCII 目录。"""
    if POSE_MODEL_CACHE.exists():
        return POSE_MODEL_CACHE

    POSE_MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if POSE_MODEL_PATH.exists():
        shutil.copy2(POSE_MODEL_PATH, POSE_MODEL_CACHE)
        return POSE_MODEL_CACHE

    print(f"正在下载姿态检测模型到 {POSE_MODEL_CACHE} ...")
    urlretrieve(POSE_MODEL_URL, POSE_MODEL_CACHE)
    print("模型下载完成")
    return POSE_MODEL_CACHE

# 本地调试模式（关闭 TCP 传输）
LOCAL_DEBUG = True
FRAME_TIMEOUT_MS = 10000
WARMUP_FRAME_COUNT = 30
DEPTH_SAMPLE_RADIUS = 2
MIN_VALID_DEPTH_M = 0.2
MAX_VALID_DEPTH_M = 5.0
NUM_KEYPOINTS = 33

ENABLE_ONE_EURO_FILTER = False
ONE_EURO_MIN_CUTOFF = 2.0
ONE_EURO_BETA = 0.02
ONE_EURO_D_CUTOFF = 1.0
KEYPOINT_CONF_THRESHOLD = 0.0

CV_WINDOW_NAME = "Motion Capture (Local Debug)"
CV_DISPLAY_WIDTH = 960
RECORDINGS_DIR = Path(__file__).resolve().parent / "recordings"

# # TCP 客户端连接配置
# CLIENT_TCP_HOST = "0.0.0.0"
# CLIENT_TCP_PORT = 65432


# class TCPBroadcastServer:
#     """基于 asyncio 的 TCP 广播服务，保持与现有客户端协议一致（4 字节长度前缀 + JSON）。"""
#
#     def __init__(self) -> None:
#         self.server: Optional[asyncio.base_events.Server] = None
#         self.clients: Set[asyncio.StreamWriter] = set()
#         self.last_framed: Optional[bytes] = None
#         self.total_frames: int = 0
#         self.start_time: float = time.time()
#         self._lock = asyncio.Lock()
#
#     async def start(self, host: str = CLIENT_TCP_HOST, port: int = CLIENT_TCP_PORT) -> None:
#         try:
#             sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#             sock.settimeout(1)
#             result = sock.connect_ex((host, port))
#             sock.close()
#
#             if result == 0:
#                 print(f"警告: 端口 {port} 已被占用，请检查是否有其他程序在使用")
#                 return
#
#             self.server = await asyncio.start_server(self._handle_client, host, port)
#             sockets = ", ".join(str(s.getsockname()) for s in self.server.sockets or [])
#             print(f"TCP 广播服务已启动: {sockets}")
#             print(f"等待客户端连接到 {host}:{port}")
#
#         except Exception as e:
#             print(f"启动TCP服务器失败: {e}")
#             raise
#
#     async def stop(self) -> None:
#         if self.server is not None:
#             self.server.close()
#             await self.server.wait_closed()
#             self.server = None
#         async with self._lock:
#             for w in list(self.clients):
#                 try:
#                     w.close()
#                 except Exception:
#                     pass
#             self.clients.clear()
#
#     async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
#         addr = writer.get_extra_info("peername")
#         print(f"新客户端尝试连接: {addr}")
#
#         try:
#             async with self._lock:
#                 self.clients.add(writer)
#                 if self.last_framed:
#                     try:
#                         writer.write(self.last_framed)
#                         await writer.drain()
#                     except Exception as e:
#                         print(f"发送初始帧失败: {e}")
#                         self.clients.discard(writer)
#                         return
#             print(f"客户端已连接: {addr} | 当前连接数: {len(self.clients)}")
#
#             while True:
#                 await asyncio.sleep(60)
#         except asyncio.CancelledError:
#             pass
#         except Exception as e:
#             print(f"客户端连接异常: {addr} - {e}")
#         finally:
#             async with self._lock:
#                 if writer in self.clients:
#                     self.clients.remove(writer)
#             try:
#                 writer.close()
#                 await writer.wait_closed()
#             except Exception:
#                 pass
#             print(f"客户端断开: {addr} | 当前连接数: {len(self.clients)}")
#
#     async def broadcast_json(self, payload: dict) -> None:
#         data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
#         framed = struct.pack(">I", len(data)) + data
#         await self.broadcast(framed)
#
#     async def broadcast(self, framed: bytes) -> None:
#         to_remove: list[asyncio.StreamWriter] = []
#         async with self._lock:
#             for w in list(self.clients):
#                 try:
#                     w.write(framed)
#                 except Exception:
#                     to_remove.append(w)
#             for w in list(self.clients):
#                 if w in to_remove:
#                     continue
#                 try:
#                     await w.drain()
#                 except Exception:
#                     to_remove.append(w)
#             for w in to_remove:
#                 try:
#                     w.close()
#                 except Exception:
#                     pass
#                 self.clients.discard(w)
#             self.last_framed = framed
#             self.total_frames += 1
#
#     def status(self) -> dict:
#         elapsed = max(1e-6, time.time() - self.start_time)
#         return {
#             "clients": len(self.clients),
#             "total_frames": self.total_frames,
#             "fps": round(self.total_frames / elapsed, 2),
#         }


class OneEuroFilter:
    def __init__(
        self,
        min_cutoff: float = ONE_EURO_MIN_CUTOFF,
        beta: float = ONE_EURO_BETA,
        d_cutoff: float = ONE_EURO_D_CUTOFF,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: Optional[float] = None
        self.dx_prev: Optional[float] = None
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self.t_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class OneEuroFilter3D:
    def __init__(self) -> None:
        self._filters = [OneEuroFilter() for _ in range(3)]
        self._last: Optional[np.ndarray] = None

    def filter(self, point: np.ndarray, t: float, update: bool) -> np.ndarray:
        if not update:
            if self._last is not None:
                return self._last.copy()
            return point.copy()

        filtered = np.array([
            self._filters[i](float(point[i]), t) for i in range(3)
        ], dtype=np.float64)
        self._last = filtered
        return filtered


class Pose3DVisualizer:
    """MediaPipe world landmarks 3D 骨架可视化（Unity 坐标约定）。

    坐标约定（髋部为原点，keypoint 数据可直接用于 Unity）：
    - +Y：头部（-Y：脚部）
    - +Z：面部朝向
    - +X：右侧

    matplotlib 3D 默认把绘图 Z 轴当作屏幕竖直方向，因此展示时将
    Unity (x, y, z) 映射为绘图坐标 (x, z, y)，使 +Y 在屏幕上向上。
    """

    def __init__(self, connections) -> None:
        plt = _plt()
        plt.ion()
        self._closed = False
        self.connections = connections
        self.fig = plt.figure("3D Pose (Unity Space)", figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        _set_axonometric_view(self.ax)
        self.fig.canvas.mpl_connect(
            "close_event", lambda _evt: setattr(self, "_closed", True))

    @staticmethod
    def _is_valid(kp: dict) -> bool:
        return kp["confidence"] > 0 and (kp["x"] != 0 or kp["y"] != 0 or kp["z"] != 0)

    @staticmethod
    def _unity_to_plot(kp: dict) -> tuple[float, float, float]:
        """Unity (x,y,z) → matplotlib 绘图 (x, z, y)。"""
        return kp["x"], kp["z"], kp["y"]

    def update(self, keypoints: list) -> None:
        if self._closed or not figure_is_alive(self.fig):
            return

        valid = {kp["id"]: kp for kp in keypoints if self._is_valid(kp)}

        self.ax.cla()
        self.ax.set_xlabel("X (m, right)")
        self.ax.set_ylabel("Z (m, +Z → face)")
        self.ax.set_zlabel("Y (m, +Y → head)")
        self.ax.set_title(f"3D Pose — Unity Space ({len(valid)}/33)")

        if valid:
            plot_pts = [self._unity_to_plot(kp) for kp in valid.values()]
            self.ax.scatter(
                [p[0] for p in plot_pts],
                [p[1] for p in plot_pts],
                [p[2] for p in plot_pts],
                c="darkorange", s=40, depthshade=True,
            )

            for conn in self.connections:
                i, j = conn.start, conn.end
                if i in valid and j in valid:
                    p1 = self._unity_to_plot(valid[i])
                    p2 = self._unity_to_plot(valid[j])
                    self.ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        color="royalblue",
                        linewidth=2,
                    )

            xs = [kp["x"] for kp in valid.values()]
            ys = [kp["y"] for kp in valid.values()]
            zs = [kp["z"] for kp in valid.values()]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            center_z = sum(zs) / len(zs)
            max_span = max(
                max(abs(v - center) for v in coords)
                for coords, center in ((xs, center_x), (ys, center_y), (zs, center_z))
            )
            half = max(0.5, max_span * 1.3)
            self.ax.set_xlim(center_x - half, center_x + half)
            self.ax.set_ylim(center_z - half, center_z + half)
            self.ax.set_zlim(center_y - half, center_y + half)
            self.ax.set_box_aspect([1, 1, 1])

        self.fig.canvas.draw_idle()

    def close(self) -> None:
        plt = _plt()
        plt.ioff()
        plt.close(self.fig)


def build_broadcast_payload(packet: dict) -> dict:
    """与 TCP 广播/录制一致：Time + 各关节角平铺为独立字段。"""
    payload: dict = {"Time": packet["timestamp"]}
    payload.update(packet.get("joint_angles", {}))
    return payload


class BroadcastRecorder:
    """将广播数据录制到本地 JSONL 文件。"""

    def __init__(self, output_dir: Path = RECORDINGS_DIR) -> None:
        self.output_dir = output_dir
        self.recording = False
        self.recorded_frames = 0
        self._file: Optional[TextIO] = None
        self._path: Optional[Path] = None

    @property
    def output_path(self) -> Optional[Path]:
        return self._path

    def start(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.output_dir / (
            f"joint_angles_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
        )
        self._file = self._path.open("w", encoding="utf-8")
        self._file.write(json.dumps({
            "type": "session_start",
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }, ensure_ascii=False) + "\n")
        self.recording = True
        self.recorded_frames = 0
        print(f"[录制] 已开始 → {self._path}")
        return self._path

    def stop(self) -> None:
        if not self.recording or self._file is None:
            return
        self._file.write(json.dumps({
            "type": "session_end",
            "ended_at": datetime.now().isoformat(timespec="seconds"),
            "total_frames": self.recorded_frames,
        }, ensure_ascii=False) + "\n")
        self._file.close()
        self._file = None
        print(f"[录制] 已停止，共 {self.recorded_frames} 帧 → {self._path}")
        self.recording = False

    def toggle(self) -> bool:
        if self.recording:
            self.stop()
        else:
            self.start()
        return self.recording

    def write(self, packet: dict) -> None:
        if not self.recording or self._file is None:
            return
        payload = build_broadcast_payload(packet)
        self._file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.recorded_frames += 1


class PoseCaptureWorker:
    """子进程：RealSense 取流 + MediaPipe 推理 + 关节角计算（无 GUI）。"""

    def __init__(self):
        self.pipeline = None
        self.pose_landmarker: Optional[PoseLandmarker] = None
        self.pose_connections = PoseLandmarksConnections.POSE_LANDMARKS
        self.frame_count = 0
        self.frame_timestamp_ms = 0
        self.start_time = time.time()
        self.camera_initialized = False
        self._frame_error_count = 0
        self._last_world_scale = 1.0
        self.one_euro_filters = [OneEuroFilter3D() for _ in range(NUM_KEYPOINTS)]
        self.joint_calculator = JointAngleCalculator()
        self.forward_kinematics = ForwardKinematics()

    @staticmethod
    def _landmark_pixel(landmark, width: int, height: int) -> tuple[int, int]:
        return int(landmark.x * width), int(landmark.y * height)

    @staticmethod
    def _world_vec(landmark) -> np.ndarray:
        return np.array([landmark.x, landmark.y, landmark.z], dtype=np.float64)

    def _sample_depth_m(self, depth_frame, x_px: int, y_px: int) -> Optional[float]:
        """在像素邻域内取深度中值，过滤无效读数。"""
        width, height = self.color_intrinsics.width, self.color_intrinsics.height
        if not (0 <= x_px < width and 0 <= y_px < height):
            return None

        samples = []
        for dy in range(-DEPTH_SAMPLE_RADIUS, DEPTH_SAMPLE_RADIUS + 1):
            for dx in range(-DEPTH_SAMPLE_RADIUS, DEPTH_SAMPLE_RADIUS + 1):
                px, py = x_px + dx, y_px + dy
                if 0 <= px < width and 0 <= py < height:
                    depth_m = depth_frame.get_distance(px, py)
                    if MIN_VALID_DEPTH_M < depth_m < MAX_VALID_DEPTH_M:
                        samples.append(depth_m)
        if not samples:
            return None
        return float(np.median(samples))

    @staticmethod
    def _world_to_unity(relative_world: np.ndarray, scale: float) -> np.ndarray:
        """MediaPipe world → Unity 坐标（髋为原点，+Y 头，-Y 脚，+Z 脸）。

        MediaPipe pose_world_landmarks：+X 右，+Y 下，+Z 朝向相机。
        """
        return np.array([
            relative_world[0] * scale,
            -relative_world[1] * scale,
            relative_world[2] * scale,
        ], dtype=np.float64)

    def _collect_depth_points_for_scale(
        self,
        landmarks_2d,
        depth_frame,
    ) -> dict[int, np.ndarray]:
        """仅用于估计 world landmarks 的真实尺度。"""
        width = self.color_intrinsics.width
        height = self.color_intrinsics.height
        depth_points: dict[int, np.ndarray] = {}

        for idx, landmark in enumerate(landmarks_2d):
            x_px, y_px = self._landmark_pixel(landmark, width, height)
            depth_m = self._sample_depth_m(depth_frame, x_px, y_px)
            if depth_m is not None:
                point = rs.rs2_deproject_pixel_to_point(
                    self.color_intrinsics, [x_px, y_px], depth_m)
                depth_points[idx] = np.array(point, dtype=np.float64)

        return depth_points

    def _estimate_world_scale(
        self,
        world_landmarks,
        depth_points: dict[int, np.ndarray],
    ) -> float:
        """用真实肩宽/髋宽与 world landmarks 的比例估计缩放系数。"""
        scale_candidates = []

        def add_pair_scale(idx_a: int, idx_b: int) -> None:
            if idx_a not in depth_points or idx_b not in depth_points:
                return
            real_dist = np.linalg.norm(depth_points[idx_a] - depth_points[idx_b])
            world_dist = np.linalg.norm(
                self._world_vec(world_landmarks[idx_a]) - self._world_vec(world_landmarks[idx_b])
            )
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
        depth_frame,
    ) -> list[dict]:
        """使用 pose_world_landmarks，输出 Unity 坐标（髋为原点，+Y 头，+Z 脸）。"""
        depth_points = self._collect_depth_points_for_scale(landmarks_2d, depth_frame)
        scale = self._estimate_world_scale(world_landmarks, depth_points)
        self._last_world_scale = scale
        hip_world = 0.5 * (
            self._world_vec(world_landmarks[PoseLandmark.LEFT_HIP])
            + self._world_vec(world_landmarks[PoseLandmark.RIGHT_HIP])
        )

        keypoints = []
        for idx, landmark in enumerate(landmarks_2d):
            relative_world = self._world_vec(world_landmarks[idx]) - hip_world
            point = self._world_to_unity(relative_world, scale)
            confidence = landmark.visibility if landmark.visibility is not None else 0.0
            keypoints.append(self._make_keypoint(idx, point, confidence, "world"))

        return keypoints

    def _apply_one_euro_filter(
        self, keypoints: list[dict], timestamp: float
    ) -> list[dict]:
        """对 MediaPipe 输出的 Unity 坐标做 One-Euro 时序滤波。"""
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

    def _start_realsense_pipeline(self) -> None:
        """启动 RealSense 并预热，确保彩色/深度流均可稳定出帧。"""
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("未检测到 RealSense 设备")

        device = devices[0]
        device_name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"检测到设备: {device_name} (SN: {serial})")

        last_error = None
        for fps in (30, 15):
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)

            try:
                print(f"正在以 {fps} FPS 启动相机流...")
                profile = self.pipeline.start(config)
                depth_sensor = profile.get_device().first_depth_sensor()
                if depth_sensor.supports(rs.option.emitter_enabled):
                    depth_sensor.set_option(rs.option.emitter_enabled, 1)
                self.depth_scale = depth_sensor.get_depth_scale()

                color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
                self.color_intrinsics = color_profile.get_intrinsics()
                print(f"相机内参: {self.color_intrinsics}")

                self._warmup_camera()
                print(f"相机流已就绪 ({fps} FPS)")
                return
            except RuntimeError as exc:
                last_error = exc
                print(f"{fps} FPS 启动失败: {exc}")
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
                self.pipeline = None

        raise RuntimeError(
            f"相机无法稳定出帧: {last_error}\n"
            "请确认:\n"
            "1. 已完全关闭 RealSense Viewer 及其他占用相机的程序\n"
            "2. 使用 USB 3.0 接口（蓝色口）直连电脑，避免 Hub\n"
            "3. 重新插拔相机后重试"
        ) from last_error

    def _warmup_camera(self) -> None:
        """丢弃启动后的不稳定帧，等待自动曝光/深度流同步。"""
        print(f"正在预热相机（丢弃前 {WARMUP_FRAME_COUNT} 帧）...")
        for i in range(WARMUP_FRAME_COUNT):
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"预热第 {i + 1}/{WARMUP_FRAME_COUNT} 帧超时: {exc}"
                ) from exc

            aligned = self.align.process(frames)
            if not aligned.get_color_frame() or not aligned.get_depth_frame():
                raise RuntimeError(
                    f"预热第 {i + 1}/{WARMUP_FRAME_COUNT} 帧缺少彩色或深度数据"
                )
        print("相机预热完成")

    def _init_pose_landmarker(self) -> None:
        """加载姿态检测模型（耗时较长，放在相机就绪之后）。"""
        print("正在加载姿态检测模型...")
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
        print("姿态检测模型加载完成")

    def initialize(self) -> None:
        """初始化相机与姿态模型。"""
        print("[采集进程] 正在初始化 RealSense 相机...")
        self.align = rs.align(rs.stream.color)
        self._start_realsense_pipeline()
        self._init_pose_landmarker()
        self.camera_initialized = True
        print("[采集进程] 相机初始化成功！")

    def shutdown(self) -> None:
        """释放相机与模型资源。"""
        if self.pipeline:
            self.pipeline.stop()
        if self.pose_landmarker:
            self.pose_landmarker.close()
        elapsed_time = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0.0
        print(f"[采集进程] 已停止: {self.frame_count} 帧 | 平均 FPS: {avg_fps:.1f}")

    def run(self, frame_queue: mp_ctx.Queue, stop_event: mp_ctx.Event) -> None:
        """采集主循环，将帧数据写入队列。"""
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
        """处理一帧，返回可序列化的帧包。"""
        if not self.camera_initialized:
            return None

        frame_start = time.time()
        self.frame_count += 1

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)

            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None

            color_image = np.asanyarray(color_frame.get_data())
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            mp_image = mediapipe.Image(
                image_format=mediapipe.ImageFormat.SRGB, data=rgb_image)
            self.frame_timestamp_ms += 33
            results = self.pose_landmarker.detect_for_video(
                mp_image, self.frame_timestamp_ms)

            timestamp = time.time()
            joint_angles: dict = {}
            keypoints: list[dict] = []

            if results.pose_landmarks and results.pose_world_landmarks:
                raw_keypoints = self._build_world_keypoints(
                    results.pose_landmarks[0],
                    results.pose_world_landmarks[0],
                    depth_frame,
                )
                keypoints = self._apply_one_euro_filter(raw_keypoints, timestamp)
                self.forward_kinematics.update_lengths(keypoints)
                joint_angles = self.joint_calculator.compute(keypoints)

                drawing_utils.draw_landmarks(
                    color_image,
                    results.pose_landmarks[0],
                    self.pose_connections,
                    drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2),
                    drawing_utils.DrawingSpec(
                        color=(255, 0, 0), thickness=2, circle_radius=2),
                )

            elapsed = time.time() - frame_start
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            valid_count = sum(1 for kp in keypoints if kp.get("source") == "world")
            cv2.putText(color_image, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(color_image, f"3D Nodes: {valid_count}/33", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(color_image, "Joint Angles", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

            self._frame_error_count = 0
            return {
                "type": "frame",
                "timestamp": timestamp,
                "frame_count": self.frame_count,
                "fps": fps,
                "joint_angles": joint_angles,
                "keypoints": keypoints,
                "color_image": color_image,
            }

        except RuntimeError as e:
            self._frame_error_count += 1
            if self._frame_error_count == 1 or self._frame_error_count % 10 == 0:
                print(f"[采集进程] 处理帧时出错: {e}")
            return None
        except Exception as e:
            print(f"[采集进程] 处理帧时出错: {e}")
            return None


class MotionCaptureApp:
    """主进程：OpenCV + Matplotlib 可视化（与采集进程隔离）。"""

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
        self.recorder = BroadcastRecorder()
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
                raw_title="MediaPipe World",
                fk_title="Joint Angles FK",
                window_title="Motion Capture 3D (RealSense)",
            )
            cv2.namedWindow(CV_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(CV_WINDOW_NAME, self._on_mouse)
            print("[主进程] 3D 对比窗口已创建（左：MediaPipe | 右：FK 重建 | 轴测视角）")
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
                1 for kp in keypoints
                if kp.get("confidence", 0) > 0
                and (kp["x"] or kp["y"] or kp["z"])
            ) if keypoints else 0
            self.visualizer.update(
                keypoints or None,
                fk_nodes,
                raw_title_suffix=f"({valid_count}/33)",
            )

    def _drain_queue(self) -> tuple[Optional[dict], bool]:
        """排空队列，返回最新帧；若收到 error 则 should_stop=True。"""
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
    """multiprocessing 子进程入口（Windows spawn 需为模块级函数）。"""
    worker = PoseCaptureWorker()
    worker.run(frame_queue, stop_event)


def main() -> None:
    ctx = mp_ctx.get_context("spawn")
    frame_queue = ctx.Queue(maxsize=2)
    stop_event = ctx.Event()

    capture_proc = ctx.Process(
        target=capture_process_entry,
        args=(frame_queue, stop_event),
        name="PoseCapture",
        daemon=True,
    )
    capture_proc.start()

    app = MotionCaptureApp(frame_queue, stop_event, capture_proc)
    app.run()


if __name__ == "__main__":
    mp_ctx.freeze_support()
    try:
        main()
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序异常退出: {e}")
