import cv2
import numpy as np
import pyrealsense2 as rs
import time
import asyncio
import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import matplotlib.pyplot as plt
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
LOCAL_DEBUG = True
FRAME_TIMEOUT_MS = 10000
WARMUP_FRAME_COUNT = 30
DEPTH_SAMPLE_RADIUS = 2
MIN_VALID_DEPTH_M = 0.2
MAX_VALID_DEPTH_M = 5.0

YOLO_MODEL_NAME = "yolo11n-pose.pt"
YOLO_CONF_THRESHOLD = 0.5
KEYPOINT_CONF_THRESHOLD = 0.5

DEPTH_JUMP_THRESHOLD_M = 0.3
BONE_LENGTH_TOLERANCE = 0.4
BONE_LENGTH_EMA_ALPHA = 0.05

ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.007
ONE_EURO_D_CUTOFF = 1.0

NUM_KEYPOINTS = 17

COCO_KEYPOINT_NAMES = [
    "NOSE", "LEFT_EYE", "RIGHT_EYE", "LEFT_EAR", "RIGHT_EAR",
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# 运动学骨骼：(parent, child)
KINEMATIC_BONES = [
    (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (5, 11), (6, 12), (5, 6), (11, 12),
    (0, 5), (0, 6),
]

LEFT_HIP = 11
RIGHT_HIP = 12


class KeypointSource(str, Enum):
    DEPTH = "depth"
    KINEMATIC = "kinematic"
    PREDICTED = "predicted"
    NONE = "none"


@dataclass
class KeypointState:
    position: np.ndarray
    confidence: float
    source: KeypointSource
    valid: bool


# ---------------------------------------------------------------------------
# One-Euro 滤波
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 骨骼运动学
# ---------------------------------------------------------------------------
class SkeletonKinematics:
    def __init__(self) -> None:
        self.bone_lengths: dict[tuple[int, int], float] = {}
        self._last_positions: Optional[list[KeypointState]] = None

    @staticmethod
    def _bone_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def update_bone_lengths(self, positions: list[KeypointState]) -> None:
        for parent, child in KINEMATIC_BONES:
            if not positions[parent].valid or not positions[child].valid:
                continue
            length = float(np.linalg.norm(
                positions[child].position - positions[parent].position))
            if length < 1e-4:
                continue
            key = self._bone_key(parent, child)
            if key not in self.bone_lengths:
                self.bone_lengths[key] = length
            else:
                old = self.bone_lengths[key]
                self.bone_lengths[key] = (
                    (1.0 - BONE_LENGTH_EMA_ALPHA) * old + BONE_LENGTH_EMA_ALPHA * length
                )

    def _expected_length(self, parent: int, child: int) -> Optional[float]:
        return self.bone_lengths.get(self._bone_key(parent, child))

    def reject_bone_outliers(self, positions: list[KeypointState]) -> None:
        for parent, child in KINEMATIC_BONES:
            if not positions[parent].valid or not positions[child].valid:
                continue
            expected = self._expected_length(parent, child)
            if expected is None:
                continue
            actual = float(np.linalg.norm(
                positions[child].position - positions[parent].position))
            if actual > expected * (1.0 + BONE_LENGTH_TOLERANCE):
                if positions[child].confidence <= positions[parent].confidence:
                    positions[child].valid = False
                    positions[child].source = KeypointSource.NONE
                else:
                    positions[parent].valid = False
                    positions[parent].source = KeypointSource.NONE

    def apply_constraints(self, positions: list[KeypointState]) -> None:
        changed = True
        while changed:
            changed = False
            for parent, child in KINEMATIC_BONES:
                p_valid = positions[parent].valid
                c_valid = positions[child].valid
                expected = self._expected_length(parent, child)
                if expected is None:
                    continue

                if p_valid and not c_valid:
                    direction = positions[child].position - positions[parent].position
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        direction /= norm
                    else:
                        direction = np.array([0.0, 1.0, 0.0])
                    positions[child].position = positions[parent].position + direction * expected
                    positions[child].valid = True
                    positions[child].source = KeypointSource.KINEMATIC
                    positions[child].confidence = positions[parent].confidence * 0.8
                    changed = True
                elif c_valid and not p_valid:
                    direction = positions[parent].position - positions[child].position
                    norm = np.linalg.norm(direction)
                    if norm > 1e-6:
                        direction /= norm
                    else:
                        direction = np.array([0.0, -1.0, 0.0])
                    positions[parent].position = positions[child].position + direction * expected
                    positions[parent].valid = True
                    positions[parent].source = KeypointSource.KINEMATIC
                    positions[parent].confidence = positions[child].confidence * 0.8
                    changed = True

    def predict_missing(self, positions: list[KeypointState]) -> None:
        if self._last_positions is None:
            return
        for i, state in enumerate(positions):
            if not state.valid and self._last_positions[i].valid:
                state.position = self._last_positions[i].position.copy()
                state.valid = True
                state.source = KeypointSource.PREDICTED
                state.confidence = self._last_positions[i].confidence * 0.5

    def commit_frame(self, positions: list[KeypointState]) -> None:
        self._last_positions = [
            KeypointState(
                pos.position.copy(), pos.confidence, pos.source, pos.valid
            )
            for pos in positions
        ]


# ---------------------------------------------------------------------------
# 3D 可视化
# ---------------------------------------------------------------------------
class Pose3DVisualizer:
    """Unity 坐标 3D 骨架（COCO 17 点）。"""

    def __init__(self, connections: list[tuple[int, int]]) -> None:
        plt.ion()
        self.connections = connections
        self.fig = plt.figure("3D Pose (YOLO Unity Space)", figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")

    @staticmethod
    def _is_valid(kp: dict) -> bool:
        return kp["confidence"] > 0 and (
            kp["x"] != 0 or kp["y"] != 0 or kp["z"] != 0
        )

    @staticmethod
    def _unity_to_plot(kp: dict) -> tuple[float, float, float]:
        return kp["x"], kp["z"], kp["y"]

    def update(self, keypoints: list) -> None:
        valid = {kp["id"]: kp for kp in keypoints if self._is_valid(kp)}

        self.ax.cla()
        self.ax.set_xlabel("X (m, right)")
        self.ax.set_ylabel("Z (m, +Z → face)")
        self.ax.set_zlabel("Y (m, +Y → head)")
        self.ax.set_title(f"3D Pose — YOLO Unity ({len(valid)}/{NUM_KEYPOINTS})")

        if valid:
            plot_pts = [self._unity_to_plot(kp) for kp in valid.values()]
            self.ax.scatter(
                [p[0] for p in plot_pts],
                [p[1] for p in plot_pts],
                [p[2] for p in plot_pts],
                c="darkorange", s=40, depthshade=True,
            )

            for i, j in self.connections:
                if i in valid and j in valid:
                    p1 = self._unity_to_plot(valid[i])
                    p2 = self._unity_to_plot(valid[j])
                    self.ax.plot(
                        [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color="royalblue", linewidth=2,
                    )

            xs = [kp["x"] for kp in valid.values()]
            ys = [kp["y"] for kp in valid.values()]
            zs = [kp["z"] for kp in valid.values()]
            center_x, center_y, center_z = (
                sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
            )
            max_span = max(
                max(abs(v - c) for v in coords)
                for coords, c in ((xs, center_x), (ys, center_y), (zs, center_z))
            )
            half = max(0.5, max_span * 1.3)
            self.ax.set_xlim(center_x - half, center_x + half)
            self.ax.set_ylim(center_z - half, center_z + half)
            self.ax.set_zlim(center_y - half, center_y + half)
            self.ax.set_box_aspect([1, 1, 1])
            self.ax.invert_zaxis()

        self.ax.view_init(elev=10, azim=-90)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)


# ---------------------------------------------------------------------------
# 2D 骨架绘制
# ---------------------------------------------------------------------------
def draw_coco_skeleton(
    image: np.ndarray,
    keypoints_xy: np.ndarray,
    confidences: np.ndarray,
    conf_threshold: float = KEYPOINT_CONF_THRESHOLD,
) -> None:
    for x, y, conf in zip(keypoints_xy[:, 0], keypoints_xy[:, 1], confidences):
        if conf >= conf_threshold:
            cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)

    for i, j in COCO_SKELETON:
        if confidences[i] >= conf_threshold and confidences[j] >= conf_threshold:
            pt1 = (int(keypoints_xy[i, 0]), int(keypoints_xy[i, 1]))
            pt2 = (int(keypoints_xy[j, 0]), int(keypoints_xy[j, 1]))
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)


# ---------------------------------------------------------------------------
# 主服务器
# ---------------------------------------------------------------------------
class MotionCaptureServer:
    """RealSense + YOLO11n-Pose + 深度 3D + 运动学 + One-Euro 滤波。"""

    def __init__(self) -> None:
        self.visualizer: Optional[Pose3DVisualizer] = None
        self.pipeline = None
        self.yolo_model: Optional[YOLO] = None
        self.kinematics = SkeletonKinematics()
        self.one_euro_filters = [OneEuroFilter3D() for _ in range(NUM_KEYPOINTS)]
        self._last_raw_depths: dict[int, float] = {}
        self.frame_count = 0
        self.start_time = time.time()
        self.camera_initialized = False
        self._frame_error_count = 0

    def _sample_depth_m(self, depth_frame, x_px: int, y_px: int) -> Optional[float]:
        width = self.color_intrinsics.width
        height = self.color_intrinsics.height
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

    def _deproject_camera(self, x_px: int, y_px: int, depth_m: float) -> np.ndarray:
        point = rs.rs2_deproject_pixel_to_point(
            self.color_intrinsics, [x_px, y_px], depth_m)
        return np.array(point, dtype=np.float64)

    @staticmethod
    def _camera_to_unity(point_cam: np.ndarray, hip_cam: np.ndarray) -> np.ndarray:
        """RealSense 相机坐标 → Unity（髋为原点，+Y 头，+Z 脸）。"""
        return np.array([
            point_cam[0] - hip_cam[0],
            hip_cam[1] - point_cam[1],
            hip_cam[2] - point_cam[2],
        ], dtype=np.float64)

    def _reject_depth_jump(self, idx: int, depth_m: float, confidence: float) -> bool:
        if idx not in self._last_raw_depths:
            return False
        if confidence >= 0.7:
            return False
        return abs(depth_m - self._last_raw_depths[idx]) > DEPTH_JUMP_THRESHOLD_M

    def _detect_yolo_pose(
        self, color_image: np.ndarray
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        results = self.yolo_model(color_image, verbose=False, conf=YOLO_CONF_THRESHOLD)
        if not results or results[0].keypoints is None:
            return None

        kpts = results[0].keypoints
        if kpts.xy is None or len(kpts.xy) == 0:
            return None

        xy = kpts.xy[0].cpu().numpy()
        conf = kpts.conf[0].cpu().numpy() if kpts.conf is not None else np.ones(len(xy))
        if xy.shape[0] < NUM_KEYPOINTS:
            return None
        return xy[:NUM_KEYPOINTS], conf[:NUM_KEYPOINTS]

    def _build_keypoints_3d(
        self,
        keypoints_xy: np.ndarray,
        confidences: np.ndarray,
        depth_frame,
        timestamp: float,
    ) -> list[dict]:
        positions = [
            KeypointState(
                position=np.zeros(3),
                confidence=float(confidences[i]),
                source=KeypointSource.NONE,
                valid=False,
            )
            for i in range(NUM_KEYPOINTS)
        ]

        cam_points: dict[int, np.ndarray] = {}
        for i in range(NUM_KEYPOINTS):
            if confidences[i] < KEYPOINT_CONF_THRESHOLD:
                continue
            x_px, y_px = int(keypoints_xy[i, 0]), int(keypoints_xy[i, 1])
            depth_m = self._sample_depth_m(depth_frame, x_px, y_px)
            if depth_m is None:
                continue
            if self._reject_depth_jump(i, depth_m, confidences[i]):
                continue
            cam_points[i] = self._deproject_camera(x_px, y_px, depth_m)
            self._last_raw_depths[i] = depth_m

        if LEFT_HIP not in cam_points or RIGHT_HIP not in cam_points:
            self.kinematics.predict_missing(positions)
            return self._finalize_keypoints(positions, timestamp)

        hip_cam = 0.5 * (cam_points[LEFT_HIP] + cam_points[RIGHT_HIP])

        for i, cam_pt in cam_points.items():
            unity_pt = self._camera_to_unity(cam_pt, hip_cam)
            positions[i].position = unity_pt
            positions[i].valid = True
            positions[i].source = KeypointSource.DEPTH

        self.kinematics.update_bone_lengths(positions)
        self.kinematics.reject_bone_outliers(positions)
        self.kinematics.apply_constraints(positions)
        self.kinematics.predict_missing(positions)
        self.kinematics.commit_frame(positions)

        return self._finalize_keypoints(positions, timestamp)

    def _finalize_keypoints(
        self, positions: list[KeypointState], timestamp: float
    ) -> list[dict]:
        keypoints = []
        for i, state in enumerate(positions):
            if state.valid:
                filtered = self.one_euro_filters[i].filter(
                    state.position, timestamp, update=True)
            else:
                filtered = self.one_euro_filters[i].filter(
                    state.position, timestamp, update=False)

            keypoints.append({
                "id": i,
                "name": COCO_KEYPOINT_NAMES[i],
                "x": float(filtered[0]),
                "y": float(filtered[1]),
                "z": float(filtered[2]),
                "confidence": float(state.confidence),
                "source": state.source.value if state.valid else KeypointSource.NONE.value,
            })
        return keypoints

    def _build_predicted_keypoints(self, timestamp: float) -> list[dict]:
        positions = [
            KeypointState(np.zeros(3), 0.0, KeypointSource.NONE, False)
            for _ in range(NUM_KEYPOINTS)
        ]
        self.kinematics.predict_missing(positions)
        return self._finalize_keypoints(positions, timestamp)

    def _start_realsense_pipeline(self) -> None:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("未检测到 RealSense 设备")

        device = devices[0]
        serial = device.get_info(rs.camera_info.serial_number)
        print(f"检测到设备: {device.get_info(rs.camera_info.name)} (SN: {serial})")

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

                color_profile = rs.video_stream_profile(
                    profile.get_stream(rs.stream.color))
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
            "请确认已关闭 RealSense Viewer 并使用 USB 3.0 直连"
        ) from last_error

    def _warmup_camera(self) -> None:
        print(f"正在预热相机（丢弃前 {WARMUP_FRAME_COUNT} 帧）...")
        for i in range(WARMUP_FRAME_COUNT):
            frames = self.pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
            aligned = self.align.process(frames)
            if not aligned.get_color_frame() or not aligned.get_depth_frame():
                raise RuntimeError(f"预热第 {i + 1}/{WARMUP_FRAME_COUNT} 帧缺少深度或彩色数据")
        print("相机预热完成")

    def _init_yolo_pose(self) -> None:
        print(f"正在加载 YOLO 姿态模型 {YOLO_MODEL_NAME} ...")
        self.yolo_model = YOLO(YOLO_MODEL_NAME)
        print("YOLO11n-Pose 模型加载完成")

    async def initialize_camera(self) -> None:
        try:
            print("正在初始化 RealSense 相机...")
            self.align = rs.align(rs.stream.color)
            self._start_realsense_pipeline()
            self._init_yolo_pose()

            if LOCAL_DEBUG:
                self.visualizer = Pose3DVisualizer(COCO_SKELETON)
                print("3D 可视化窗口已创建（YOLO / Unity）")

            self.camera_initialized = True
            print("相机初始化成功！")
        except Exception as e:
            print(f"相机初始化失败: {e}")
            if self.pipeline:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass
            raise

    async def start(self) -> None:
        await self.initialize_camera()
        print("YOLO 本地调试模式已启动...")

    async def stop(self) -> None:
        if self.visualizer:
            self.visualizer.close()
        if self.pipeline:
            self.pipeline.stop()
        cv2.destroyAllWindows()
        elapsed = time.time() - self.start_time
        fps_avg = self.frame_count / elapsed if elapsed > 0 else 0
        print(f"服务器已停止: {self.frame_count} 帧 | 平均 FPS: {fps_avg:.1f}")

    def process_frame(self):
        if not self.camera_initialized:
            return None

        frame_start = time.time()
        self.frame_count += 1
        timestamp = time.time()

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=FRAME_TIMEOUT_MS)
            aligned = self.align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                return None

            color_image = np.asanyarray(color_frame.get_data())
            detection = self._detect_yolo_pose(color_image)

            if detection is not None:
                keypoints_xy, confidences = detection
                keypoints = self._build_keypoints_3d(
                    keypoints_xy, confidences, depth_frame, timestamp)
            else:
                keypoints_xy, confidences = None, None
                keypoints = self._build_predicted_keypoints(timestamp)

            fps = 1.0 / max(time.time() - frame_start, 1e-6)
            data = {
                "timestamp": timestamp,
                "frame_count": self.frame_count,
                "fps": fps,
                "keypoints": keypoints,
            }
            self._frame_error_count = 0
            return data, color_image, keypoints_xy, confidences

        except RuntimeError as e:
            self._frame_error_count += 1
            if self._frame_error_count == 1 or self._frame_error_count % 10 == 0:
                print(f"处理帧时出错: {e}")
            return None
        except Exception as e:
            print(f"处理帧时出错: {e}")
            return None

    async def run_capture_loop(self) -> None:
        try:
            while True:
                result = self.process_frame()
                if result is None:
                    await asyncio.sleep(0.1)
                    continue

                data, color_image, keypoints_xy, confidences = result

                if keypoints_xy is not None and confidences is not None:
                    draw_coco_skeleton(color_image, keypoints_xy, confidences)

                valid_count = sum(
                    1 for kp in data["keypoints"]
                    if kp["source"] != KeypointSource.NONE.value
                )
                cv2.putText(color_image, f"FPS: {data['fps']:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(color_image, f"3D Nodes: {valid_count}/{NUM_KEYPOINTS}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(color_image, "YOLO11n UNITY", (10, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 255), 2)

                cv2.imshow("Motion Capture (YOLO)", color_image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if self.visualizer and data["keypoints"]:
                    self.visualizer.update(data["keypoints"])

                await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            print("正在关闭服务器...")
        finally:
            await self.stop()


async def main() -> None:
    try:
        server = MotionCaptureServer()
        await server.start()
        await server.run_capture_loop()
    except Exception as e:
        print(f"服务器运行出错: {e}")
        print("请检查相机连接与 ultralytics 安装")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"程序异常退出: {e}")
