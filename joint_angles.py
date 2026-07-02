"""关节角计算（MediaPipe / ZED Unity 关键点）与正向运动学重建。

角度定义与符号约定见项目根目录可对照的 description.json；
计算流程参照 localaxes_to_jointangles.py：相邻骨段旋转矩阵 → 欧拉角。
3 DOF 关节的绕骨轴 twist 通过段坐标系构造规则锁死；Elbow_AZ、Wrist_AY、
Ankle_AY 锁长轴自旋；Wrist_AZ 锁侧向。有手/脚标记时用真实点，否则回退近似。
所有角度单位为弧度。
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — 注册 3d 投影

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

_plt_module = None

# MediaPipe PoseLandmark 索引（33 点）
MP = {
    "LS": 11, "RS": 12, "LE": 13, "RE": 14, "LW": 15, "RW": 16,
    "LP": 17, "RP": 18, "LI": 19, "RI": 20, "LT": 21, "RT": 22,
    "LH": 23, "RH": 24, "LK": 25, "RK": 26, "LA": 27, "RA": 28,
    "LHE": 29, "RHE": 30, "LFI": 31, "RFI": 32,
}

JOINT_ANGLE_KEYS = [
    "Hip_rAX", "Hip_rAY", "Hip_rAZ", "Hip_lAX", "Hip_lAY", "Hip_lAZ",
    "Knee_rAX", "Knee_rAY", "Knee_rAZ", "Knee_lAX", "Knee_lAY", "Knee_lAZ",
    "Ankle_rAX", "Ankle_rAY", "Ankle_rAZ", "Ankle_lAX", "Ankle_lAY", "Ankle_lAZ",
    "Arm_rAX", "Arm_rAY", "Arm_rAZ", "Arm_lAX", "Arm_lAY", "Arm_lAZ",
    "Elbow_rAX", "Elbow_rAY", "Elbow_rAZ", "Elbow_lAX", "Elbow_lAY", "Elbow_lAZ",
    "Wrist_rAX", "Wrist_rAY", "Wrist_rAZ", "Wrist_lAX", "Wrist_lAY", "Wrist_lAZ",
    "Pelvis_AX", "Pelvis_AY", "Pelvis_AZ",
]

FK_CONNECTIONS = [
    ("pelvis", "hip_l"), ("pelvis", "hip_r"),
    ("hip_l", "knee_l"), ("knee_l", "ankle_l"), ("ankle_l", "foot_l"),
    ("hip_r", "knee_r"), ("knee_r", "ankle_r"), ("ankle_r", "foot_r"),
    ("pelvis", "shoulder_l"), ("pelvis", "shoulder_r"),
    ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"), ("wrist_l", "hand_l"),
    ("shoulder_r", "elbow_r"), ("elbow_r", "wrist_r"), ("wrist_r", "hand_r"),
]

# 手/脚段默认骨长（米）；明显短于前臂/小腿，避免 FK 可视化过长
DEFAULT_SEGMENT_LENGTHS: dict[str, float] = {
    "hand_l": 0.10,
    "hand_r": 0.10,
    "foot_l": 0.12,
    "foot_r": 0.12,
}

# 脚尖/指端最低置信度；低于此或几何异常时回退默认姿态
DISTAL_MIN_CONF = 0.5
# |dot(足段, 小腿)| 超过此值视为「与腿平行」的坏点
FOOT_SHANK_PARALLEL_COS = 0.85

# 轴测（二测）默认视角：同时看到 X/Y/Z 三个方向
AXONOMETRIC_ELEV = 35.0
AXONOMETRIC_AZIM = 45.0


def _plt():
    """延迟加载 matplotlib；子进程使用 Agg 后端，避免与 OpenCV GUI 争用 GIL。"""
    global _plt_module
    if _plt_module is not None:
        return _plt_module

    import matplotlib
    backend = os.environ.get("MPLBACKEND", matplotlib.get_backend()).lower()
    if backend == "agg":
        import matplotlib.pyplot as plt
    else:
        if not backend.endswith("agg"):
            matplotlib.use("TkAgg", force=False)
        import matplotlib.pyplot as plt
    _plt_module = plt
    return plt


def figure_is_alive(fig) -> bool:
    return _plt().fignum_exists(fig.number)


def present_figure(fig) -> None:
    """立即重绘 matplotlib 窗口（draw + flush_events，不用 plt.pause）。"""
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if backend == "agg" or backend.endswith("agg"):
        return
    if fig is None or not figure_is_alive(fig):
        return
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
    except Exception:
        pass


def pump_matplotlib_events(figs: Optional[list] = None) -> None:
    """刷新 matplotlib 窗口事件（仅 flush_events，不用 plt.pause）。

    plt.pause 会在拖动 3D 窗口时嵌套跑 Tk 事件循环并释放 GIL，
    随后 cv2.waitKey 触发 PyEval_RestoreThread 崩溃。
    实际重绘由 present_figure() 在 update 时完成。
    """
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if backend == "agg" or backend.endswith("agg"):
        return
    if not figs:
        return
    for fig in figs:
        if fig is None or not figure_is_alive(fig):
            continue
        try:
            fig.canvas.flush_events()
        except Exception:
            pass


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return v / n


def _kps_array(keypoints: list[dict]) -> dict[int, np.ndarray]:
    return {
        kp["id"]: np.array([kp["x"], kp["y"], kp["z"]], dtype=np.float64)
        for kp in keypoints
    }


def _kp_confidence(keypoints: list[dict]) -> dict[int, float]:
    return {kp["id"]: float(kp.get("confidence", 0.0)) for kp in keypoints}


def _point_valid(
    pts: dict[int, np.ndarray],
    idx: int,
    conf: Optional[dict[int, float]] = None,
    *,
    min_conf: float = 0.01,
) -> bool:
    if idx not in pts:
        return False
    p = pts[idx]
    if not np.all(np.isfinite(p)) or np.linalg.norm(p) < 1e-5:
        return False
    if conf is not None and conf.get(idx, 0.0) < min_conf:
        return False
    return True


def _hand_landmark_ids(side: str) -> tuple[int, int, int]:
    """食指、拇指、小指（MediaPipe 顺序）。"""
    if side == "l":
        return MP["LI"], MP["LT"], MP["LP"]
    return MP["RI"], MP["RT"], MP["RP"]


def _foot_landmark_ids(side: str) -> tuple[int, int]:
    if side == "l":
        return MP["LFI"], MP["LHE"]
    return MP["RFI"], MP["RHE"]


def _hand_distal(pts: dict[int, np.ndarray], side: str, conf: dict[int, float]) -> Optional[np.ndarray]:
    """掌段远端：优先食指，其次拇指/小指，或可用点均值。"""
    avail = [
        pts[idx] for idx in _hand_landmark_ids(side)
        if _point_valid(pts, idx, conf)
    ]
    if not avail:
        return None
    if _point_valid(pts, _hand_landmark_ids(side)[0], conf):
        return pts[_hand_landmark_ids(side)[0]]
    return np.mean(avail, axis=0)


def _hand_rotation(
    wrist: np.ndarray,
    R_fore: np.ndarray,
    pts: dict[int, np.ndarray],
    side: str,
    conf: dict[int, float],
) -> np.ndarray:
    distal = _hand_distal(pts, side, conf)
    if distal is None:
        return R_fore
    twist_ref = R_fore[:, 2]
    return _segment_rotation(wrist, distal, twist_ref)


def _foot_rotation(
    ankle: np.ndarray,
    pts: dict[int, np.ndarray],
    side: str,
    conf: dict[int, float],
    twist_ref: np.ndarray,
    knee: np.ndarray,
    R_pelvis: np.ndarray,
) -> np.ndarray:
    """足段世界姿态。无可靠脚尖时相对小腿无旋转（角度≈0），FK 可视化再补默认脚板。"""
    R_shank = _segment_rotation(knee, ankle, twist_ref)
    toe_id, heel_id = _foot_landmark_ids(side)
    shank_axis = R_shank[:, 1]
    if _point_valid(pts, toe_id, conf, min_conf=DISTAL_MIN_CONF):
        toe = pts[toe_id]
        toe_vec = toe - ankle
        if np.linalg.norm(toe_vec) > 1e-5:
            toe_dir = _normalize(toe_vec)
            if abs(float(np.dot(toe_dir, shank_axis))) < FOOT_SHANK_PARALLEL_COS:
                foot_twist = twist_ref
                if _point_valid(pts, heel_id, conf, min_conf=DISTAL_MIN_CONF):
                    heel_vec = pts[heel_id] - ankle
                    if np.linalg.norm(heel_vec) > 1e-5:
                        foot_twist = heel_vec
                return _segment_rotation(ankle, toe, foot_twist)
    return R_shank


def _global_axes_in_unity() -> np.ndarray:
    """绝对坐标系 → Unity 坐标（+X 右，+Y 上，+Z 脸），右手系 det=+1。

    description.json：X=前后，Y=垂直，Z=左右。
    e_z = e_x × e_y → Unity 中 global Z 对应 -X。
    """
    ex = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ey = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ez = np.cross(ex, ey)
    return np.column_stack([ex, ey, ez])


def _build_pelvis_frame(pts: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """骨盆段旋转矩阵（列向量为 x/y/z 轴，Unity 世界系）。"""
    lh, rh = pts[MP["LH"]], pts[MP["RH"]]
    ls, rs = pts[MP["LS"]], pts[MP["RS"]]
    origin = 0.5 * (lh + rh)
    x_axis = _normalize(rh - lh)
    mid_shoulder = 0.5 * (ls + rs)
    y_hint = _normalize(mid_shoulder - origin)
    z_axis = _normalize(np.cross(x_axis, y_hint))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    return origin, np.column_stack([x_axis, y_axis, z_axis])


def _segment_rotation(
    proximal: np.ndarray,
    distal: np.ndarray,
    twist_ref: np.ndarray,
) -> np.ndarray:
    """由两点 + 参考向量构造段姿态（锁 twist：y 沿骨轴，z 为 twist_ref 在法平面投影）。"""
    y = _normalize(distal - proximal)
    if np.linalg.norm(y) < 1e-8:
        return np.eye(3, dtype=np.float64)

    z = twist_ref - np.dot(twist_ref, y) * y
    if np.linalg.norm(z) < 1e-6:
        alt = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(y[2])) > 0.9:
            alt = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        z = alt - np.dot(alt, y) * y
    z = _normalize(z)
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def _relative_rotation(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    return parent.T @ child


def _safe_rotation(R: np.ndarray) -> np.ndarray:
    """SVD 投影到最近的合法旋转矩阵（det=+1）。"""
    u, _, vt = np.linalg.svd(R)
    r = u @ vt
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vt
    return r


def _euler(seq: str, R: np.ndarray) -> tuple[float, float, float]:
    from scipy.spatial.transform import Rotation as SciR
    return tuple(float(v) for v in SciR.from_matrix(_safe_rotation(R)).as_euler(seq, degrees=False))


def _from_euler(seq: str, angles: tuple[float, float, float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation as SciR
    return SciR.from_euler(seq, angles, degrees=False).as_matrix()


def _decode_zxy_limb(
    ax: float, ay: float, az: float, side: str,
) -> tuple[float, float, float]:
    """髋/膝：localaxes ZXY 编码的逆映射。"""
    if side == "l":
        return ax, az, -ay
    return ax, -az, ay


def _encode_zxy_limb(
    rot_z: float, rot_x: float, rot_y: float, side: str,
) -> tuple[float, float, float]:
    """髋/膝：ZXY 欧拉 → description 字段。"""
    if side == "l":
        return rot_z, -rot_y, rot_x
    return rot_z, rot_y, -rot_x


def _encode_zxy_ankle(
    rot_z: float, rot_x: float, _rot_y: float, side: str,
) -> tuple[float, float, float]:
    if side == "l":
        return rot_z, 0.0, rot_x
    return rot_z, 0.0, -rot_x


def _decode_zxy_ankle(ax: float, _ay: float, az: float, side: str) -> tuple[float, float, float]:
    if side == "l":
        return ax, az, 0.0
    return ax, -az, 0.0


def _encode_yxz_arm(
    rot_y: float, rot_x: float, rot_z: float, side: str,
) -> tuple[float, float, float]:
    """肩：localaxes YXZ；AY 为第二角（锁 twist 时通常≈0）。"""
    if side == "l":
        return rot_z, -rot_x, rot_y
    return rot_z, rot_x, rot_y


def _decode_yxz_arm(ax: float, ay: float, az: float, side: str) -> tuple[float, float, float]:
    if side == "l":
        return az, -ay, ax
    return az, ay, ax


def _encode_zxy_elbow(
    rot_z: float, rot_x: float, rot_y: float, side: str,
) -> tuple[float, float, float]:
    """肘：localaxes ZXY；AZ 为第三角（应≈0），AX/AY 为屈伸/旋转。"""
    if side == "l":
        return rot_z, -rot_x, rot_y
    return -rot_z, rot_x, rot_y


def _decode_zxy_elbow(ax: float, ay: float, az: float, side: str) -> tuple[float, float, float]:
    if side == "l":
        return ax, -ay, az
    return -ax, ay, az


def _default_foot_rotation(
    ankle: np.ndarray,
    knee: np.ndarray,
    R_pelvis: np.ndarray,
) -> np.ndarray:
    """无可靠脚尖时：脚板水平指向前方（+Z），与小腿近似 90°。"""
    shank = _normalize(ankle - knee)
    forward = R_pelvis[:, 2]
    foot_dir = forward - np.dot(forward, shank) * shank
    if np.linalg.norm(foot_dir) < 1e-6:
        lateral = R_pelvis[:, 0]
        foot_dir = lateral - np.dot(lateral, shank) * shank
    foot_dir = _normalize(foot_dir)
    up = R_pelvis[:, 1]
    return _segment_rotation(ankle, ankle + foot_dir, up)


def _approx_foot_rotation(
    knee: np.ndarray,
    ankle: np.ndarray,
    R_pelvis: np.ndarray,
) -> np.ndarray:
    return _default_foot_rotation(ankle, knee, R_pelvis)


def _pelvis_angles(R_pelvis: np.ndarray) -> tuple[float, float, float]:
    """骨盆相对绝对坐标系，XZY 顺序（同 localaxes_to_jointangles）。"""
    G = _global_axes_in_unity()
    R_rel = _relative_rotation(G, R_pelvis)
    az, ax, ay = _euler("XZY", R_rel)
    return ax, ay, az


def _pelvis_rotation(ax: float, ay: float, az: float) -> np.ndarray:
    G = _global_axes_in_unity()
    R_rel = _from_euler("XZY", (az, ax, ay))
    return G @ R_rel


class JointAngleCalculator:
    """从 Unity 关键点计算关节角（弧度，定义见 description.json）。"""

    def compute(self, keypoints: list[dict]) -> dict[str, float]:
        pts = _kps_array(keypoints)
        conf = _kp_confidence(keypoints)
        _origin, R_pelvis = _build_pelvis_frame(pts)
        twist_ref = R_pelvis[:, 2]
        angles: dict[str, float] = {k: 0.0 for k in JOINT_ANGLE_KEYS}

        px, py, pz = _pelvis_angles(R_pelvis)
        angles["Pelvis_AX"] = px
        angles["Pelvis_AY"] = py
        angles["Pelvis_AZ"] = pz

        for side, hip_id, knee_id, ankle_id, shoulder_id, elbow_id, wrist_id in (
            ("r", MP["RH"], MP["RK"], MP["RA"], MP["RS"], MP["RE"], MP["RW"]),
            ("l", MP["LH"], MP["LK"], MP["LA"], MP["LS"], MP["LE"], MP["LW"]),
        ):
            hip, knee, ankle = pts[hip_id], pts[knee_id], pts[ankle_id]
            shoulder, elbow, wrist = pts[shoulder_id], pts[elbow_id], pts[wrist_id]

            R_thigh = _segment_rotation(hip, knee, twist_ref)
            R_hip = _relative_rotation(R_pelvis, R_thigh)
            h_ax, h_ay, h_az = _encode_zxy_limb(*_euler("ZXY", R_hip), side)
            angles[f"Hip_{side}AX"] = h_ax
            angles[f"Hip_{side}AY"] = h_ay
            angles[f"Hip_{side}AZ"] = h_az

            R_shank = _segment_rotation(knee, ankle, twist_ref)
            R_knee = _relative_rotation(R_thigh, R_shank)
            k_ax, k_ay, k_az = _encode_zxy_limb(*_euler("ZXY", R_knee), side)
            angles[f"Knee_{side}AX"] = k_ax
            angles[f"Knee_{side}AY"] = k_ay
            angles[f"Knee_{side}AZ"] = k_az

            R_foot = _foot_rotation(
                ankle, pts, side, conf, twist_ref, knee, R_pelvis,
            )
            R_ankle = _relative_rotation(R_shank, R_foot)
            a_ax, a_ay, a_az = _encode_zxy_ankle(*_euler("ZXY", R_ankle), side)
            angles[f"Ankle_{side}AX"] = a_ax
            angles[f"Ankle_{side}AY"] = a_ay
            angles[f"Ankle_{side}AZ"] = a_az

            R_upper = _segment_rotation(shoulder, elbow, twist_ref)
            R_shoulder = _relative_rotation(R_pelvis, R_upper)
            s_ax, s_ay, s_az = _encode_yxz_arm(*_euler("YXZ", R_shoulder), side)
            angles[f"Arm_{side}AX"] = s_ax
            angles[f"Arm_{side}AY"] = s_ay
            angles[f"Arm_{side}AZ"] = s_az

            R_fore = _segment_rotation(elbow, wrist, twist_ref)
            R_elbow = _relative_rotation(R_upper, R_fore)
            e_ax, e_ay, e_az = _encode_zxy_elbow(*_euler("ZXY", R_elbow), side)
            angles[f"Elbow_{side}AX"] = e_ax
            angles[f"Elbow_{side}AY"] = e_ay
            angles[f"Elbow_{side}AZ"] = e_az

            R_hand = _hand_rotation(wrist, R_fore, pts, side, conf)
            R_wrist = _relative_rotation(R_fore, R_hand)
            w_ax, w_ay, w_az = _encode_zxy_ankle(*_euler("ZXY", R_wrist), side)
            angles[f"Wrist_{side}AX"] = w_ax
            angles[f"Wrist_{side}AY"] = w_ay
            angles[f"Wrist_{side}AZ"] = w_az

        return angles


class ForwardKinematics:
    """由关节角正向运动学重建骨架节点（Unity 坐标，角度为弧度）。"""

    def __init__(self) -> None:
        self.segment_lengths: dict[str, float] = {}
        self.anchor_offsets: dict[str, np.ndarray] = {}

    def update_lengths(self, keypoints: list[dict]) -> None:
        pts = _kps_array(keypoints)
        conf = _kp_confidence(keypoints)
        origin, R = _build_pelvis_frame(pts)
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
            "hand_l": (MP["LW"], MP["LI"]),
            "hand_r": (MP["RW"], MP["RI"]),
            "foot_l": (MP["LA"], MP["LFI"]),
            "foot_r": (MP["RA"], MP["RFI"]),
        }
        for name, (a, b) in pairs.items():
            if name.startswith(("foot_", "hand_")):
                if not (_point_valid(pts, a, conf) and _point_valid(pts, b, conf, min_conf=DISTAL_MIN_CONF)):
                    continue
            length = float(np.linalg.norm(pts[b] - pts[a]))
            if length > 1e-4:
                if name.startswith("foot_"):
                    if name.endswith("_l"):
                        shank_a, shank_b = MP["LK"], MP["LA"]
                        shank_key = "shank_l"
                    else:
                        shank_a, shank_b = MP["RK"], MP["RA"]
                        shank_key = "shank_r"
                    shank_len = float(np.linalg.norm(pts[shank_b] - pts[shank_a]))
                    if shank_len < 1e-4:
                        shank_len = self.segment_lengths.get(shank_key, 0.40)
                    # 拒绝异常长段（坏点拉到原点），保留正常足长 ~6–22 cm
                    length = min(length, max(0.50 * shank_len, 0.10), 0.22)
                    length = max(length, 0.06)
                if name not in self.segment_lengths:
                    self.segment_lengths[name] = length
                else:
                    self.segment_lengths[name] = (
                        0.9 * self.segment_lengths[name] + 0.1 * length
                    )

    def _L(self, name: str, default: float | None = None) -> float:
        if default is None:
            default = DEFAULT_SEGMENT_LENGTHS.get(name, 0.25)
        return self.segment_lengths.get(name, default)

    def _anchor(self, name: str, default: np.ndarray) -> np.ndarray:
        return self.anchor_offsets.get(name, default)

    def rebuild(self, angles: dict[str, float]) -> dict[str, np.ndarray]:
        pos = np.zeros(3, dtype=np.float64)
        R_pelvis = _pelvis_rotation(
            angles["Pelvis_AX"], angles["Pelvis_AY"], angles["Pelvis_AZ"],
        )

        nodes: dict[str, np.ndarray] = {"pelvis": pos.copy()}
        nodes["hip_l"] = pos + R_pelvis @ self._anchor("hip_l", np.array([-0.08, 0.0, 0.0]))
        nodes["hip_r"] = pos + R_pelvis @ self._anchor("hip_r", np.array([0.08, 0.0, 0.0]))
        nodes["shoulder_l"] = pos + R_pelvis @ self._anchor(
            "shoulder_l", np.array([-0.08, 0.35, 0.0]))
        nodes["shoulder_r"] = pos + R_pelvis @ self._anchor(
            "shoulder_r", np.array([0.08, 0.35, 0.0]))

        def leg_chain(parent: str, side: str, seg1: str, seg2: str, prefix: str) -> None:
            base = nodes[parent]
            hip_e = _decode_zxy_limb(
                angles[f"{prefix}AX"], angles[f"{prefix}AY"], angles[f"{prefix}AZ"], side,
            )
            R_thigh = R_pelvis @ _from_euler("ZXY", hip_e)
            thigh_dir = R_thigh[:, 1]
            knee = base + thigh_dir * self._L(seg1)
            nodes[f"knee_{side}"] = knee

            knee_e = _decode_zxy_limb(
                angles[f"Knee_{side}AX"],
                angles[f"Knee_{side}AY"],
                angles[f"Knee_{side}AZ"],
                side,
            )
            R_shank = R_thigh @ _from_euler("ZXY", knee_e)
            ankle = knee + R_shank[:, 1] * self._L(seg2)
            nodes[f"ankle_{side}"] = ankle

            ankle_e = _decode_zxy_ankle(
                angles[f"Ankle_{side}AX"],
                angles[f"Ankle_{side}AY"],
                angles[f"Ankle_{side}AZ"],
                side,
            )
            R_foot = R_shank @ _from_euler("ZXY", ankle_e)
            foot_dir = R_foot[:, 1]
            if abs(float(np.dot(foot_dir, R_shank[:, 1]))) >= FOOT_SHANK_PARALLEL_COS:
                R_foot = _default_foot_rotation(ankle, knee, R_pelvis)
            nodes[f"foot_{side}"] = ankle + R_foot[:, 1] * self._L(f"foot_{side}")

        leg_chain("hip_l", "l", "thigh_l", "shank_l", "Hip_l")
        leg_chain("hip_r", "r", "thigh_r", "shank_r", "Hip_r")

        def arm_chain(side: str) -> None:
            shoulder = nodes[f"shoulder_{side}"]
            arm_e = _decode_yxz_arm(
                angles[f"Arm_{side}AX"],
                angles[f"Arm_{side}AY"],
                angles[f"Arm_{side}AZ"],
                side,
            )
            R_upper = R_pelvis @ _from_euler("YXZ", arm_e)
            elbow = shoulder + R_upper[:, 1] * self._L(f"upper_{side}")
            nodes[f"elbow_{side}"] = elbow

            elbow_e = _decode_zxy_elbow(
                angles[f"Elbow_{side}AX"],
                angles[f"Elbow_{side}AY"],
                angles[f"Elbow_{side}AZ"],
                side,
            )
            R_fore = R_upper @ _from_euler("ZXY", elbow_e)
            wrist = elbow + R_fore[:, 1] * self._L(f"fore_{side}")
            nodes[f"wrist_{side}"] = wrist

            wrist_e = _decode_zxy_ankle(
                angles[f"Wrist_{side}AX"],
                angles[f"Wrist_{side}AY"],
                angles[f"Wrist_{side}AZ"],
                side,
            )
            R_hand = R_fore @ _from_euler("ZXY", wrist_e)
            nodes[f"hand_{side}"] = wrist + R_hand[:, 1] * self._L(f"hand_{side}")

        arm_chain("l")
        arm_chain("r")
        return nodes


def _connection_indices(conn) -> tuple[int, int]:
    if hasattr(conn, "start"):
        return int(conn.start), int(conn.end)
    return int(conn[0]), int(conn[1])


def _set_axonometric_view(ax, elev: float = AXONOMETRIC_ELEV, azim: float = AXONOMETRIC_AZIM) -> None:
    ax.view_init(elev=elev, azim=azim)


def _style_pose_axis(
    ax,
    title: str,
    *,
    elev: float = AXONOMETRIC_ELEV,
    azim: float = AXONOMETRIC_AZIM,
) -> None:
    ax.set_xlabel("X (m, right)")
    ax.set_ylabel("Z (m, +Z → face)")
    ax.set_zlabel("Y (m, +Y → head)")
    ax.set_title(title)
    _set_axonometric_view(ax, elev, azim)


def _fit_axis_limits(
    ax,
    plot_pts: list[tuple[float, float, float]],
    *,
    zoom: float = 1.0,
) -> None:
    if not plot_pts:
        return
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
    half = max(0.5, span * 1.3) / max(zoom, 0.1)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_zlim(cz - half, cz + half)
    ax.set_box_aspect([1, 1, 1])


class CombinedPose3DVisualizer:
    """左：原始关键点；右：关节角 FK 重建。Agg 离屏渲染后由 OpenCV 显示。"""

    def __init__(
        self,
        raw_connections,
        *,
        raw_kp_valid=None,
        raw_to_plot=None,
        fk_to_plot=None,
        fk_connections=None,
        raw_title: str = "Raw Keypoints",
        fk_title: str = "FK Reconstructed",
        window_title: str = "3D Pose Comparison",
        cv_window: Optional[str] = None,
    ) -> None:
        self._closed = False
        self._cv_window = cv_window or window_title
        self.raw_connections = raw_connections
        self.fk_connections = fk_connections or FK_CONNECTIONS
        self.raw_kp_valid = raw_kp_valid or (
            lambda kp: kp.get("confidence", 0) > 0
            and (kp["x"] != 0 or kp["y"] != 0 or kp["z"] != 0)
        )
        self.raw_to_plot = raw_to_plot or (
            lambda kp: (float(kp["x"]), float(kp["z"]), float(kp["y"]))
        )
        self.fk_to_plot = fk_to_plot or (
            lambda p: (float(p[0]), float(p[2]), float(p[1]))
        )
        self.raw_title = raw_title
        self.fk_title = fk_title
        self._elev = AXONOMETRIC_ELEV
        self._azim = AXONOMETRIC_AZIM
        self._zoom = 1.0
        self._dragging = False
        self._last_mouse: Optional[tuple[int, int]] = None
        self._cached_raw: Optional[list[dict]] = None
        self._cached_raw_suffix = ""
        self._cached_fk: Optional[dict[str, np.ndarray]] = None

        self._fig = Figure(figsize=(14, 7), dpi=100)
        self._canvas = FigureCanvasAgg(self._fig)
        self.ax_raw = self._fig.add_subplot(1, 2, 1, projection="3d")
        self.ax_fk = self._fig.add_subplot(1, 2, 2, projection="3d")
        self._fig.subplots_adjust(left=0.02, right=0.98, wspace=0.08)
        self._style_axis(self.ax_raw, self.raw_title)
        self._style_axis(self.ax_fk, self.fk_title)
        _fit_axis_limits(self.ax_raw, [(0.0, 0.0, 0.0)], zoom=self._zoom)
        _fit_axis_limits(self.ax_fk, [(0.0, 0.0, 0.0)], zoom=self._zoom)

        cv2.namedWindow(self._cv_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._cv_window, 1400, 700)
        cv2.setMouseCallback(self._cv_window, self._on_mouse)
        self._present()

    @property
    def fig(self):
        return self._fig

    def _style_axis(self, ax, title: str) -> None:
        _style_pose_axis(ax, title, elev=self._elev, azim=self._azim)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param) -> None:
        if self._closed:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._dragging = True
            self._last_mouse = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._dragging = False
            self._last_mouse = None
        elif event == cv2.EVENT_MOUSEMOVE and self._dragging and self._last_mouse:
            dx = x - self._last_mouse[0]
            dy = y - self._last_mouse[1]
            self._azim = (self._azim + dx * 0.4) % 360.0
            self._elev = float(np.clip(self._elev - dy * 0.4, -89.0, 89.0))
            self._last_mouse = (x, y)
            self._redraw()
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags) if hasattr(cv2, "getMouseWheelDelta") else flags
            if delta > 0:
                self._zoom = min(4.0, self._zoom * 1.08)
            elif delta < 0:
                self._zoom = max(0.25, self._zoom / 1.08)
            self._redraw()
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            self._elev = AXONOMETRIC_ELEV
            self._azim = AXONOMETRIC_AZIM
            self._zoom = 1.0
            self._redraw()

    def _present(self) -> None:
        if self._closed:
            return
        self._canvas.draw()
        rgba = np.asarray(self._canvas.buffer_rgba())
        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        cv2.imshow(self._cv_window, bgr)

    def _draw_raw(self, ax, keypoints: list[dict], title_suffix: str = "") -> None:
        ax.clear()
        title = self.raw_title + (f" {title_suffix}" if title_suffix else "")
        self._style_axis(ax, title)

        valid = {kp["id"]: kp for kp in keypoints if self.raw_kp_valid(kp)}
        if not valid:
            return

        plot_pts = [self.raw_to_plot(kp) for kp in valid.values()]
        ax.scatter(
            [p[0] for p in plot_pts], [p[1] for p in plot_pts], [p[2] for p in plot_pts],
            c="darkorange", s=36, depthshade=True,
        )
        for conn in self.raw_connections:
            i, j = _connection_indices(conn)
            if i in valid and j in valid:
                p1 = self.raw_to_plot(valid[i])
                p2 = self.raw_to_plot(valid[j])
                ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color="royalblue", linewidth=2,
                )

        _fit_axis_limits(ax, plot_pts, zoom=self._zoom)

    def _draw_fk(self, ax, nodes: dict[str, np.ndarray]) -> None:
        ax.clear()
        self._style_axis(ax, self.fk_title)

        if not nodes:
            return

        plot_pts = [self.fk_to_plot(p) for p in nodes.values()]
        ax.scatter(
            [p[0] for p in plot_pts], [p[1] for p in plot_pts], [p[2] for p in plot_pts],
            c="mediumseagreen", s=36, depthshade=True,
        )
        for a, b in self.fk_connections:
            if a in nodes and b in nodes:
                p1 = self.fk_to_plot(nodes[a])
                p2 = self.fk_to_plot(nodes[b])
                ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color="royalblue", linewidth=2,
                )

        _fit_axis_limits(ax, plot_pts, zoom=self._zoom)

    def _redraw(self) -> None:
        if self._cached_raw is not None:
            self._draw_raw(self.ax_raw, self._cached_raw, self._cached_raw_suffix)
        if self._cached_fk is not None:
            self._draw_fk(self.ax_fk, self._cached_fk)
        self._present()

    def update(
        self,
        raw_keypoints: Optional[list[dict]] = None,
        fk_nodes: Optional[dict[str, np.ndarray]] = None,
        raw_title_suffix: str = "",
    ) -> None:
        if self._closed:
            return
        if raw_keypoints is not None:
            self._cached_raw = raw_keypoints
            self._cached_raw_suffix = raw_title_suffix
        if fk_nodes is not None:
            self._cached_fk = fk_nodes
        self._redraw()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            cv2.setMouseCallback(self._cv_window, lambda *_args: None)
            cv2.destroyWindow(self._cv_window)
        except cv2.error:
            pass
        self._fig.clf()


class ReconstructedPoseVisualizer:
    """由关节角 FK 重建的 3D 骨架窗口（与原始关键点窗口对比）。"""

    def __init__(self) -> None:
        plt = _plt()
        plt.ion()
        self._closed = False
        self.fig = plt.figure("3D Pose (Joint Angles FK)", figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        _set_axonometric_view(self.ax)
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
            present_figure(self.fig)
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
        present_figure(self.fig)

    def close(self) -> None:
        plt = _plt()
        plt.ioff()
        plt.close(self.fig)
