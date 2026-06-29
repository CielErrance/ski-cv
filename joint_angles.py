"""关节角计算（由 MediaPipe Unity 关键点）与正向运动学重建。"""

from __future__ import annotations

import math
from typing import Optional

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np


def figure_is_alive(fig: plt.Figure) -> bool:
    return plt.fignum_exists(fig.number)


def pump_matplotlib_events() -> None:
    """统一处理 matplotlib GUI 事件，避免与 OpenCV waitKey 冲突。"""
    plt.pause(0.001)

# MediaPipe PoseLandmark 索引
MP = {
    "LS": 11, "RS": 12, "LE": 13, "RE": 14, "LW": 15, "RW": 16,
    "LH": 23, "RH": 24, "LK": 25, "RK": 26, "LA": 27, "RA": 28,
}

JOINT_ANGLE_KEYS = [
    "Hip_rAX", "Hip_rAY", "Hip_rAZ", "Hip_lAX", "Hip_lAY", "Hip_lAZ",
    "Knee_rAX", "Knee_rAY", "Knee_rAZ", "Knee_lAX", "Knee_lAY", "Knee_lAZ",
    "Ankle_rAX", "Ankle_rAZ", "Ankle_lAX", "Ankle_lAZ",
    "Arm_rAX", "Arm_rAY", "Arm_rAZ", "Arm_lAX", "Arm_lAY", "Arm_lAZ",
    "Elbow_rAX", "Elbow_rAY", "Elbow_lAX", "Elbow_lAY",
    "Wrist_rAX", "Wrist_lAX",
    "Pelvis_AX", "Pelvis_AY", "Pelvis_AZ",
    "Pelvis_DX", "Pelvis_DY", "Pelvis_DZ",
]

FK_CONNECTIONS = [
    ("pelvis", "hip_l"), ("pelvis", "hip_r"),
    ("hip_l", "knee_l"), ("knee_l", "ankle_l"),
    ("hip_r", "knee_r"), ("knee_r", "ankle_r"),
    ("pelvis", "shoulder_l"), ("pelvis", "shoulder_r"),
    ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"),
    ("shoulder_r", "elbow_r"), ("elbow_r", "wrist_r"),
]


def _deg(rad: float) -> float:
    return float(np.degrees(rad))


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros(3)
    return v / n


def _kps_array(keypoints: list[dict]) -> dict[int, np.ndarray]:
    return {kp["id"]: np.array([kp["x"], kp["y"], kp["z"]], dtype=np.float64)
            for kp in keypoints}


def _rotation_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rotation_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rotation_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def _euler_xyz(R: np.ndarray) -> tuple[float, float, float]:
    """旋转矩阵 → XYZ 欧拉角（弧度）。"""
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        ax = math.atan2(R[2, 1], R[2, 2])
        ay = math.atan2(-R[2, 0], sy)
        az = math.atan2(R[1, 0], R[0, 0])
    else:
        ax = math.atan2(-R[1, 2], R[1, 1])
        ay = math.atan2(-R[2, 0], sy)
        az = 0.0
    return ax, ay, az


def _build_body_frame(pts: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """骨盆局部坐标系：+X 右，+Y 上，+Z 脸。"""
    lh, rh = pts[MP["LH"]], pts[MP["RH"]]
    ls, rs = pts[MP["LS"]], pts[MP["RS"]]
    origin = 0.5 * (lh + rh)
    x_axis = _normalize(rh - lh)
    mid_shoulder = 0.5 * (ls + rs)
    y_hint = _normalize(mid_shoulder - origin)
    z_axis = _normalize(np.cross(x_axis, y_hint))
    y_axis = np.cross(z_axis, x_axis)
    y_axis = _normalize(y_axis)
    R = np.column_stack([x_axis, y_axis, z_axis])
    return origin, R


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    cos_v = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return _deg(math.acos(cos_v))


def _flexion_angle(parent: np.ndarray, joint: np.ndarray, child: np.ndarray) -> float:
    """关节屈曲角（度）：伸直=0，弯曲为正值。"""
    proximal = _normalize(parent - joint)
    distal = _normalize(child - joint)
    return max(0.0, 180.0 - _angle_between(proximal, distal))


def _extension_angle(parent: np.ndarray, joint: np.ndarray, child: np.ndarray) -> float:
    """关节伸展角（度）：伸直=0，屈曲为负值。"""
    proximal = _normalize(parent - joint)
    distal = _normalize(child - joint)
    return _angle_between(proximal, distal) - 180.0


_BONE_REST = np.array([0.0, -1.0, 0.0], dtype=np.float64)


def _spherical_dir(theta: float, phi: float) -> np.ndarray:
    return np.array([
        math.sin(theta) * math.cos(phi),
        math.cos(theta),
        math.sin(theta) * math.sin(phi),
    ], dtype=np.float64)


def _rotate_vector(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis)
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return _normalize(v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1.0 - c))


def _hip_angles_from_local(d: np.ndarray, side: str) -> tuple[float, float, float]:
    if side == "r":
        ax = _deg(math.atan2(d[2], d[1]))
        ay = _deg(math.atan2(-d[0], math.hypot(d[1], d[2])))
        az = _deg(math.atan2(d[0], d[1]))
    else:
        ax = _deg(math.atan2(d[2], d[1]))
        ay = _deg(math.atan2(d[0], math.hypot(d[1], d[2])))
        az = _deg(math.atan2(-d[0], d[1]))
    return ax, ay, az


def _arm_angles_from_local(d: np.ndarray, side: str) -> tuple[float, float, float]:
    if side == "r":
        ax = _deg(math.atan2(-d[0], d[1]))
        ay = _deg(math.atan2(d[0], d[2]))
        az = _deg(math.atan2(d[2], d[1]))
    else:
        ax = _deg(math.atan2(d[0], d[1]))
        ay = _deg(math.atan2(-d[0], d[2]))
        az = _deg(math.atan2(d[2], d[1]))
    return ax, ay, az


def _solve_unit_dir_from_angles(
    predict_fn,
    ax: float,
    ay: float,
    az: float,
    side: str,
) -> np.ndarray:
    """在球面上搜索与给定关节角最一致的单位方向。"""
    best_dir = _BONE_REST.copy()
    best_loss = 1e18
    for ti in range(37):
        theta = ti * math.pi / 36.0
        for pi in range(73):
            phi = pi * 2.0 * math.pi / 72.0
            d = _spherical_dir(theta, phi)
            pa, pb, pc = predict_fn(d, side)
            loss = (pa - ax) ** 2 + (pb - ay) ** 2 + (pc - az) ** 2
            if loss < best_loss:
                best_loss = loss
                best_dir = d
    return _normalize(best_dir)


def _hip_dir_local(ax: float, ay: float, az: float, side: str) -> np.ndarray:
    return _solve_unit_dir_from_angles(_hip_angles_from_local, ax, ay, az, side)


def _arm_dir_local(ax: float, ay: float, az: float, side: str) -> np.ndarray:
    return _solve_unit_dir_from_angles(_arm_angles_from_local, ax, ay, az, side)


def _elbow_frame(upper_dir: np.ndarray, R_body: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """肘/膝处局部坐标：y=近端骨，x=屈伸轴，z=完成右手系。"""
    y = _normalize(upper_dir)
    z = R_body[:, 2] - np.dot(R_body[:, 2], y) * y
    if np.linalg.norm(z) < 1e-6:
        z = R_body[:, 0] - np.dot(R_body[:, 0], y) * y
    z = _normalize(z)
    x = _normalize(np.cross(y, z))
    return x, y, z


def _elbow_forearm_dir(
    upper_dir: np.ndarray,
    R_body: np.ndarray,
    flex_deg: float,
    ay_deg: float,
    side: str,
) -> np.ndarray:
    """由肘角重建前臂方向（与 _elbow_angles 互逆）。"""
    x_axis, y_axis, _z_axis = _elbow_frame(upper_dir, R_body)
    fore = _rotate_vector(y_axis, x_axis, math.radians(flex_deg))
    best_dir = fore
    best_err = 1e9
    for step in range(72):
        twist = step * 5.0
        candidate = _rotate_vector(fore, y_axis, math.radians(twist))
        local = R_body.T @ candidate
        pred = (
            _deg(math.atan2(local[0], local[1]))
            if side == "r"
            else _deg(math.atan2(-local[0], local[1]))
        )
        err = abs((pred - ay_deg + 180.0) % 360.0 - 180.0)
        if err < best_err:
            best_err = err
            best_dir = candidate
    return _normalize(best_dir)


def _knee_shank_dir(
    thigh_dir: np.ndarray,
    R_body: np.ndarray,
    ext_deg: float,
    ay_deg: float,
    az_deg: float,
    side: str,
) -> np.ndarray:
    """由膝角重建小腿方向。"""
    x_axis, y_axis, _z_axis = _elbow_frame(thigh_dir, R_body)
    shank = _rotate_vector(y_axis, x_axis, math.radians(-ext_deg))
    shank = _rotate_vector(shank, y_axis, math.radians(az_deg))
    best_dir = shank
    best_err = 1e9
    for step in range(72):
        twist = step * 5.0
        candidate = _rotate_vector(shank, y_axis, math.radians(twist))
        local = R_body.T @ candidate
        pred = (
            _deg(math.atan2(-local[0], local[1]))
            if side == "r"
            else _deg(math.atan2(local[0], local[1]))
        )
        err = abs((pred - ay_deg + 180.0) % 360.0 - 180.0)
        if err < best_err:
            best_err = err
            best_dir = candidate
    return _normalize(best_dir)


class JointAngleCalculator:
    """从 Unity 关键点计算图中定义的关节角（度）。"""

    def __init__(self) -> None:
        self.pelvis_reference: Optional[np.ndarray] = None

    def _build_body_frame(self, pts: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        return _build_body_frame(pts)

    def _local_dir(self, R: np.ndarray, origin: np.ndarray, target: np.ndarray) -> np.ndarray:
        return _normalize(R.T @ (target - origin))

    def _hip_angles(self, R: np.ndarray, hip: np.ndarray, knee: np.ndarray, side: str) -> tuple[float, float, float]:
        d = self._local_dir(R, hip, knee)
        if side == "r":
            # AX +前 -后, AY +内 -外, AZ +内旋 -外旋
            ax = _deg(math.atan2(d[2], d[1]))
            ay = _deg(math.atan2(-d[0], math.hypot(d[1], d[2])))
            az = _deg(math.atan2(d[0], d[1]))
        else:
            ax = _deg(math.atan2(d[2], d[1]))
            ay = _deg(math.atan2(d[0], math.hypot(d[1], d[2])))
            az = _deg(math.atan2(-d[0], d[1]))
        return ax, ay, az

    def _knee_angles(
        self, R: np.ndarray, hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray, side: str
    ) -> tuple[float, float, float]:
        ax = _extension_angle(hip, knee, ankle)  # +伸 -屈，直腿=0
        shank_local = self._local_dir(R, knee, ankle)
        ay = _deg(math.atan2(-shank_local[0], shank_local[1])) if side == "r" else _deg(
            math.atan2(shank_local[0], shank_local[1]))
        az = _deg(math.atan2(shank_local[0], shank_local[2]))
        return ax, ay, az

    def _ankle_angles(
        self, R: np.ndarray, knee: np.ndarray, ankle: np.ndarray, side: str
    ) -> tuple[float, float]:
        shank = _normalize(knee - ankle)
        foot_hint = _normalize(R[:, 2])  # 朝 +Z（面部）为参考
        ax = _angle_between(shank, foot_hint) - 90.0  # +背屈 -跖屈
        local = self._local_dir(R, ankle, ankle + foot_hint)
        az = _deg(math.atan2(local[0], local[2])) if side == "r" else _deg(math.atan2(-local[0], local[2]))
        return ax, az

    def _arm_angles(
        self, R: np.ndarray, shoulder: np.ndarray, elbow: np.ndarray, side: str
    ) -> tuple[float, float, float]:
        d = self._local_dir(R, shoulder, elbow)
        if side == "r":
            ax = _deg(math.atan2(-d[0], d[1]))
            ay = _deg(math.atan2(d[0], d[2]))
            az = _deg(math.atan2(d[2], d[1]))
        else:
            ax = _deg(math.atan2(d[0], d[1]))
            ay = _deg(math.atan2(-d[0], d[2]))
            az = _deg(math.atan2(d[2], d[1]))
        return ax, ay, az

    def _elbow_angles(
        self, R: np.ndarray, shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray, side: str
    ) -> tuple[float, float]:
        ax = _flexion_angle(shoulder, elbow, wrist)  # +屈肘，直臂=0
        fore_local = self._local_dir(R, elbow, wrist)
        ay = _deg(math.atan2(fore_local[0], fore_local[1])) if side == "r" else _deg(
            math.atan2(-fore_local[0], fore_local[1]))
        return ax, ay

    def _wrist_angles(
        self, R: np.ndarray, elbow: np.ndarray, wrist: np.ndarray, side: str
    ) -> float:
        fore = _normalize(wrist - elbow)
        ref = _normalize(R[:, 0])
        ax = _angle_between(fore, ref) - 90.0
        return ax if side == "r" else -ax

    def compute(
        self,
        keypoints: list[dict],
        pelvis_position: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        pts = _kps_array(keypoints)
        origin, R_body = self._build_body_frame(pts)
        ax_p, ay_p, az_p = _euler_xyz(R_body)
        angles: dict[str, float] = {k: 0.0 for k in JOINT_ANGLE_KEYS}

        if pelvis_position is None:
            pelvis_position = origin
        if self.pelvis_reference is None:
            self.pelvis_reference = pelvis_position.copy()
        delta = pelvis_position - self.pelvis_reference

        angles["Pelvis_AX"] = _deg(ax_p)
        angles["Pelvis_AY"] = _deg(ay_p)
        angles["Pelvis_AZ"] = _deg(az_p)
        angles["Pelvis_DX"] = float(delta[0])
        angles["Pelvis_DY"] = float(delta[1])
        angles["Pelvis_DZ"] = float(delta[2])

        hr_ax, hr_ay, hr_az = self._hip_angles(R_body, pts[MP["RH"]], pts[MP["RK"]], "r")
        hl_ax, hl_ay, hl_az = self._hip_angles(R_body, pts[MP["LH"]], pts[MP["LK"]], "l")
        angles["Hip_rAX"], angles["Hip_rAY"], angles["Hip_rAZ"] = hr_ax, hr_ay, hr_az
        angles["Hip_lAX"], angles["Hip_lAY"], angles["Hip_lAZ"] = hl_ax, hl_ay, hl_az

        kr = self._knee_angles(R_body, pts[MP["RH"]], pts[MP["RK"]], pts[MP["RA"]], "r")
        kl = self._knee_angles(R_body, pts[MP["LH"]], pts[MP["LK"]], pts[MP["LA"]], "l")
        angles["Knee_rAX"], angles["Knee_rAY"], angles["Knee_rAZ"] = kr
        angles["Knee_lAX"], angles["Knee_lAY"], angles["Knee_lAZ"] = kl

        ar_ax, ar_az = self._ankle_angles(R_body, pts[MP["RK"]], pts[MP["RA"]], "r")
        al_ax, al_az = self._ankle_angles(R_body, pts[MP["LK"]], pts[MP["LA"]], "l")
        angles["Ankle_rAX"], angles["Ankle_rAZ"] = ar_ax, ar_az
        angles["Ankle_lAX"], angles["Ankle_lAZ"] = al_ax, al_az

        sr = self._arm_angles(R_body, pts[MP["RS"]], pts[MP["RE"]], "r")
        sl = self._arm_angles(R_body, pts[MP["LS"]], pts[MP["LE"]], "l")
        angles["Arm_rAX"], angles["Arm_rAY"], angles["Arm_rAZ"] = sr
        angles["Arm_lAX"], angles["Arm_lAY"], angles["Arm_lAZ"] = sl

        er = self._elbow_angles(R_body, pts[MP["RS"]], pts[MP["RE"]], pts[MP["RW"]], "r")
        el = self._elbow_angles(R_body, pts[MP["LS"]], pts[MP["LE"]], pts[MP["LW"]], "l")
        angles["Elbow_rAX"], angles["Elbow_rAY"] = er
        angles["Elbow_lAX"], angles["Elbow_lAY"] = el

        angles["Wrist_rAX"] = self._wrist_angles(R_body, pts[MP["RE"]], pts[MP["RW"]], "r")
        angles["Wrist_lAX"] = self._wrist_angles(R_body, pts[MP["LE"]], pts[MP["LW"]], "l")

        return angles


class ForwardKinematics:
    """由关节角正向运动学重建骨架节点（Unity 坐标）。"""

    def __init__(self) -> None:
        self.segment_lengths: dict[str, float] = {}
        self.anchor_offsets: dict[str, np.ndarray] = {}

    def update_lengths(self, keypoints: list[dict]) -> None:
        pts = _kps_array(keypoints)
        origin, R = _build_body_frame(pts)
        for name, idx in {
            "hip_l": MP["LH"],
            "hip_r": MP["RH"],
            "shoulder_l": MP["LS"],
            "shoulder_r": MP["RS"],
        }.items():
            local = R.T @ (pts[idx] - origin)
            if np.linalg.norm(local) > 1e-4:
                if name not in self.anchor_offsets:
                    self.anchor_offsets[name] = local
                else:
                    self.anchor_offsets[name] = (
                        0.9 * self.anchor_offsets[name] + 0.1 * local
                    )

        pairs = {
            "thigh_l": (MP["LH"], MP["LK"]),
            "thigh_r": (MP["RH"], MP["RK"]),
            "shank_l": (MP["LK"], MP["LA"]),
            "shank_r": (MP["RK"], MP["RA"]),
            "upper_l": (MP["LS"], MP["LE"]),
            "upper_r": (MP["RS"], MP["RE"]),
            "fore_l": (MP["LE"], MP["LW"]),
            "fore_r": (MP["RE"], MP["RW"]),
        }
        for name, (a, b) in pairs.items():
            length = float(np.linalg.norm(pts[b] - pts[a]))
            if length > 1e-4:
                if name not in self.segment_lengths:
                    self.segment_lengths[name] = length
                else:
                    self.segment_lengths[name] = 0.9 * self.segment_lengths[name] + 0.1 * length

    def _L(self, name: str, default: float = 0.25) -> float:
        return self.segment_lengths.get(name, default)

    def _anchor(self, name: str, default: np.ndarray) -> np.ndarray:
        return self.anchor_offsets.get(name, default)

    def rebuild(self, angles: dict[str, float]) -> dict[str, np.ndarray]:
        pos = np.array([
            angles["Pelvis_DX"], angles["Pelvis_DY"], angles["Pelvis_DZ"]
        ], dtype=np.float64)
        R = (
            _rotation_z(math.radians(angles["Pelvis_AZ"]))
            @ _rotation_y(math.radians(angles["Pelvis_AY"]))
            @ _rotation_x(math.radians(angles["Pelvis_AX"]))
        )

        nodes: dict[str, np.ndarray] = {"pelvis": pos.copy()}
        nodes["hip_l"] = pos + R @ self._anchor("hip_l", np.array([-0.08, 0.0, 0.0]))
        nodes["hip_r"] = pos + R @ self._anchor("hip_r", np.array([0.08, 0.0, 0.0]))
        nodes["shoulder_l"] = pos + R @ self._anchor(
            "shoulder_l", np.array([-0.08, 0.35, 0.0]))
        nodes["shoulder_r"] = pos + R @ self._anchor(
            "shoulder_r", np.array([0.08, 0.35, 0.0]))

        def leg_chain(parent: str, side: str, seg1: str, seg2: str, prefix: str) -> None:
            base = nodes[parent]
            hip_local = _hip_dir_local(
                angles[f"{prefix}AX"], angles[f"{prefix}AY"], angles[f"{prefix}AZ"], side)
            thigh_dir = _normalize(R @ hip_local)
            knee = base + thigh_dir * self._L(seg1)
            nodes[f"knee_{side}"] = knee
            shank_dir = _knee_shank_dir(
                thigh_dir, R,
                angles[f"Knee_{side}AX"],
                angles[f"Knee_{side}AY"],
                angles[f"Knee_{side}AZ"],
                side,
            )
            nodes[f"ankle_{side}"] = knee + shank_dir * self._L(seg2)

        leg_chain("hip_l", "l", "thigh_l", "shank_l", "Hip_l")
        leg_chain("hip_r", "r", "thigh_r", "shank_r", "Hip_r")

        def arm_chain(side: str) -> None:
            shoulder = nodes[f"shoulder_{side}"]
            arm_local = _arm_dir_local(
                angles[f"Arm_{side}AX"],
                angles[f"Arm_{side}AY"],
                angles[f"Arm_{side}AZ"],
                side,
            )
            upper_dir = _normalize(R @ arm_local)
            elbow = shoulder + upper_dir * self._L(f"upper_{side}")
            nodes[f"elbow_{side}"] = elbow
            fore_dir = _elbow_forearm_dir(
                upper_dir, R,
                angles[f"Elbow_{side}AX"],
                angles[f"Elbow_{side}AY"],
                side,
            )
            nodes[f"wrist_{side}"] = elbow + fore_dir * self._L(f"fore_{side}")

        arm_chain("l")
        arm_chain("r")
        return nodes


class ReconstructedPoseVisualizer:
    """由关节角 FK 重建的 3D 骨架窗口（与原始关键点窗口对比）。"""

    def __init__(self) -> None:
        plt.ion()
        self._closed = False
        self.fig = plt.figure("3D Pose (Joint Angles FK)", figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.view_init(elev=10, azim=-90)
        self.fig.canvas.mpl_connect(
            "close_event", lambda _evt: setattr(self, "_closed", True))

    @staticmethod
    def _unity_to_plot(p: np.ndarray) -> tuple[float, float, float]:
        return float(p[0]), float(p[2]), float(p[1])

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

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        cx, cy, cz = sum(xs) / len(xs), sum(ys) / len(ys), sum(zs) / len(zs)
        span = max(
            max(abs(v - c) for v in coords)
            for coords, c in ((xs, cx), (ys, cy), (zs, cz))
        )
        half = max(0.5, span * 1.3)
        self.ax.set_xlim(cx - half, cx + half)
        self.ax.set_ylim(cz - half, cz + half)
        self.ax.set_zlim(cy - half, cy + half)
        self.ax.set_box_aspect([1, 1, 1])
        self.ax.invert_zaxis()
        self.fig.canvas.draw_idle()

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)
