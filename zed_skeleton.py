"""ZED BODY_34 骨架 → MediaPipe 兼容关键点 / Unity 坐标。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from joint_angles import MP

NUM_MP_KEYPOINTS = 33
NUM_ZED_KEYPOINTS = 34

# ZED BODY_34 索引（Stereolabs 文档）
ZED = {
    "PELVIS": 0,
    "NAVAL_SPINE": 1,
    "CHEST_SPINE": 2,
    "NECK": 3,
    "LEFT_CLAVICLE": 4,
    "LEFT_SHOULDER": 5,
    "LEFT_ELBOW": 6,
    "LEFT_WRIST": 7,
    "LEFT_HAND": 8,
    "LEFT_HANDTIP": 9,
    "LEFT_THUMB": 10,
    "RIGHT_CLAVICLE": 11,
    "RIGHT_SHOULDER": 12,
    "RIGHT_ELBOW": 13,
    "RIGHT_WRIST": 14,
    "RIGHT_HAND": 15,
    "RIGHT_HANDTIP": 16,
    "RIGHT_THUMB": 17,
    "LEFT_HIP": 18,
    "LEFT_KNEE": 19,
    "LEFT_ANKLE": 20,
    "LEFT_FOOT": 21,
    "RIGHT_HIP": 22,
    "RIGHT_KNEE": 23,
    "RIGHT_ANKLE": 24,
    "RIGHT_FOOT": 25,
    "HEAD": 26,
    "NOSE": 27,
    "LEFT_EYE": 28,
    "LEFT_EAR": 29,
    "RIGHT_EYE": 30,
    "RIGHT_EAR": 31,
    "LEFT_HEEL": 32,
    "RIGHT_HEEL": 33,
}

ZED_PART_NAMES = {idx: name for name, idx in ZED.items()}

ZED_TO_MP: dict[int, int] = {
    ZED["LEFT_SHOULDER"]: MP["LS"],
    ZED["RIGHT_SHOULDER"]: MP["RS"],
    ZED["LEFT_ELBOW"]: MP["LE"],
    ZED["RIGHT_ELBOW"]: MP["RE"],
    ZED["LEFT_WRIST"]: MP["LW"],
    ZED["RIGHT_WRIST"]: MP["RW"],
    ZED["LEFT_HANDTIP"]: MP["LI"],
    ZED["RIGHT_HANDTIP"]: MP["RI"],
    ZED["LEFT_THUMB"]: MP["LT"],
    ZED["RIGHT_THUMB"]: MP["RT"],
    ZED["LEFT_HAND"]: MP["LP"],
    ZED["RIGHT_HAND"]: MP["RP"],
    ZED["LEFT_HIP"]: MP["LH"],
    ZED["RIGHT_HIP"]: MP["RH"],
    ZED["LEFT_KNEE"]: MP["LK"],
    ZED["RIGHT_KNEE"]: MP["RK"],
    ZED["LEFT_ANKLE"]: MP["LA"],
    ZED["RIGHT_ANKLE"]: MP["RA"],
    ZED["LEFT_FOOT"]: MP["LFI"],
    ZED["RIGHT_FOOT"]: MP["RFI"],
    ZED["LEFT_HEEL"]: MP["LHE"],
    ZED["RIGHT_HEEL"]: MP["RHE"],
}

MP_NAMES = {
    MP["LS"]: "LEFT_SHOULDER",
    MP["RS"]: "RIGHT_SHOULDER",
    MP["LE"]: "LEFT_ELBOW",
    MP["RE"]: "RIGHT_ELBOW",
    MP["LW"]: "LEFT_WRIST",
    MP["RW"]: "RIGHT_WRIST",
    MP["LP"]: "LEFT_PINKY",
    MP["RP"]: "RIGHT_PINKY",
    MP["LI"]: "LEFT_INDEX",
    MP["RI"]: "RIGHT_INDEX",
    MP["LT"]: "LEFT_THUMB",
    MP["RT"]: "RIGHT_THUMB",
    MP["LH"]: "LEFT_HIP",
    MP["RH"]: "RIGHT_HIP",
    MP["LK"]: "LEFT_KNEE",
    MP["RK"]: "RIGHT_KNEE",
    MP["LA"]: "LEFT_ANKLE",
    MP["RA"]: "RIGHT_ANKLE",
    MP["LHE"]: "LEFT_HEEL",
    MP["RHE"]: "RIGHT_HEEL",
    MP["LFI"]: "LEFT_FOOT_INDEX",
    MP["RFI"]: "RIGHT_FOOT_INDEX",
}

# sl::BODY_34_BONES（Camera.hpp）
BODY_34_BONES: list[tuple[int, int]] = [
    (ZED["PELVIS"], ZED["NAVAL_SPINE"]),
    (ZED["NAVAL_SPINE"], ZED["CHEST_SPINE"]),
    (ZED["CHEST_SPINE"], ZED["LEFT_CLAVICLE"]),
    (ZED["LEFT_CLAVICLE"], ZED["LEFT_SHOULDER"]),
    (ZED["LEFT_SHOULDER"], ZED["LEFT_ELBOW"]),
    (ZED["LEFT_ELBOW"], ZED["LEFT_WRIST"]),
    (ZED["LEFT_WRIST"], ZED["LEFT_HAND"]),
    (ZED["LEFT_HAND"], ZED["LEFT_HANDTIP"]),
    (ZED["LEFT_WRIST"], ZED["LEFT_THUMB"]),
    (ZED["CHEST_SPINE"], ZED["RIGHT_CLAVICLE"]),
    (ZED["RIGHT_CLAVICLE"], ZED["RIGHT_SHOULDER"]),
    (ZED["RIGHT_SHOULDER"], ZED["RIGHT_ELBOW"]),
    (ZED["RIGHT_ELBOW"], ZED["RIGHT_WRIST"]),
    (ZED["RIGHT_WRIST"], ZED["RIGHT_HAND"]),
    (ZED["RIGHT_HAND"], ZED["RIGHT_HANDTIP"]),
    (ZED["RIGHT_WRIST"], ZED["RIGHT_THUMB"]),
    (ZED["PELVIS"], ZED["LEFT_HIP"]),
    (ZED["LEFT_HIP"], ZED["LEFT_KNEE"]),
    (ZED["LEFT_KNEE"], ZED["LEFT_ANKLE"]),
    (ZED["LEFT_ANKLE"], ZED["LEFT_FOOT"]),
    (ZED["PELVIS"], ZED["RIGHT_HIP"]),
    (ZED["RIGHT_HIP"], ZED["RIGHT_KNEE"]),
    (ZED["RIGHT_KNEE"], ZED["RIGHT_ANKLE"]),
    (ZED["RIGHT_ANKLE"], ZED["RIGHT_FOOT"]),
    (ZED["CHEST_SPINE"], ZED["NECK"]),
    (ZED["NECK"], ZED["HEAD"]),
    (ZED["HEAD"], ZED["NOSE"]),
    (ZED["NOSE"], ZED["LEFT_EYE"]),
    (ZED["LEFT_EYE"], ZED["LEFT_EAR"]),
    (ZED["NOSE"], ZED["RIGHT_EYE"]),
    (ZED["RIGHT_EYE"], ZED["RIGHT_EAR"]),
    (ZED["LEFT_ANKLE"], ZED["LEFT_HEEL"]),
    (ZED["RIGHT_ANKLE"], ZED["RIGHT_HEEL"]),
    (ZED["LEFT_HEEL"], ZED["LEFT_FOOT"]),
    (ZED["RIGHT_HEEL"], ZED["RIGHT_FOOT"]),
]


@dataclass
class SkeletonConnection:
    start: int
    end: int


def zed_body34_connections() -> list[SkeletonConnection]:
    """3D/2D 可视化用 ZED BODY_34 全骨架骨段。"""
    return [SkeletonConnection(a, b) for a, b in BODY_34_BONES]


def mp_pose_connections() -> list[SkeletonConnection]:
    """关节角 FK 对比用 MediaPipe 索引骨段（保留兼容）。"""
    return [
        SkeletonConnection(MP["LS"], MP["LE"]),
        SkeletonConnection(MP["LE"], MP["LW"]),
        SkeletonConnection(MP["RS"], MP["RE"]),
        SkeletonConnection(MP["RE"], MP["RW"]),
        SkeletonConnection(MP["LH"], MP["LK"]),
        SkeletonConnection(MP["LK"], MP["LA"]),
        SkeletonConnection(MP["RH"], MP["RK"]),
        SkeletonConnection(MP["RK"], MP["RA"]),
        SkeletonConnection(MP["LS"], MP["LH"]),
        SkeletonConnection(MP["RS"], MP["RH"]),
    ]


def zed_body_connections() -> list[tuple[int, int]]:
    """2D 叠加用 ZED BODY_34 全骨架。"""
    return BODY_34_BONES


def _read_keypoint_3d(body: Any, zed_idx: int) -> np.ndarray:
    kp = body.keypoint[zed_idx]
    return np.array([float(kp[0]), float(kp[1]), float(kp[2])], dtype=np.float64)


def _read_confidence(body: Any, zed_idx: int) -> float:
    if hasattr(body, "keypoint_confidence"):
        return float(body.keypoint_confidence[zed_idx])
    if hasattr(body, "keypoint_confidence_2d"):
        return float(body.keypoint_confidence_2d[zed_idx])
    return 1.0


def _zed_kp3d_valid(point: np.ndarray) -> bool:
    """与 ZED SDK 示例一致：坐标有限即视为有效。"""
    return bool(np.isfinite(point[0]))


def zed_world_to_unity_absolute(point_world: np.ndarray) -> np.ndarray:
    """ZED WORLD（Y 上）→ Unity 约定（+Y 头，+Z 脸）。"""
    return np.array([point_world[0], point_world[1], point_world[2]], dtype=np.float64)


def _hip_center(body: Any) -> Optional[np.ndarray]:
    lh = _read_keypoint_3d(body, ZED["LEFT_HIP"])
    rh = _read_keypoint_3d(body, ZED["RIGHT_HIP"])
    if _zed_kp3d_valid(lh) and _zed_kp3d_valid(rh):
        return 0.5 * (lh + rh)
    pelvis = _read_keypoint_3d(body, ZED["PELVIS"])
    if _zed_kp3d_valid(pelvis):
        return pelvis
    chest = _read_keypoint_3d(body, ZED["CHEST_SPINE"])
    if _zed_kp3d_valid(chest):
        return chest
    return None


def zed_all_keypoints_to_unity(body: Any) -> list[dict]:
    """ZED BodyData → 全部 34 点 Unity 坐标（髋为原点），供 3D 可视化。"""
    hip_world = _hip_center(body)
    if hip_world is None:
        return []

    keypoints: list[dict] = []

    for zed_idx in range(NUM_ZED_KEYPOINTS):
        pt = _read_keypoint_3d(body, zed_idx)
        confidence = _read_confidence(body, zed_idx)
        if not _zed_kp3d_valid(pt):
            confidence = 0.0
            relative = np.zeros(3, dtype=np.float64)
            source = "none"
        else:
            relative = zed_world_to_unity_absolute(pt - hip_world)
            if confidence <= 0:
                confidence = 1.0
            source = "zed"

        keypoints.append({
            "id": zed_idx,
            "name": ZED_PART_NAMES.get(zed_idx, f"ZED_{zed_idx}"),
            "x": float(relative[0]),
            "y": float(relative[1]),
            "z": float(relative[2]),
            "confidence": confidence,
            "source": source,
        })

    return keypoints


def zed_viz_keypoints_to_unity(body: Any) -> list[dict]:
    """34 点 Unity 坐标（与关节角/FK 同一约定），供 3D 可视化。"""
    return zed_all_keypoints_to_unity(body)


def zed_keypoints_to_unity(body: Any) -> list[dict]:
    """ZED BodyData → MediaPipe 兼容 33 点（躯干/肢段 + 手/脚标记），供关节角计算。"""
    hip_world = _hip_center(body)
    if hip_world is None:
        return []

    raw: dict[int, np.ndarray] = {}
    conf: dict[int, float] = {}
    for zed_idx, mp_idx in ZED_TO_MP.items():
        pt = _read_keypoint_3d(body, zed_idx)
        confidence = _read_confidence(body, zed_idx)
        if not _zed_kp3d_valid(pt):
            confidence = 0.0
        else:
            raw[mp_idx] = pt
            conf[mp_idx] = confidence if confidence > 0 else 1.0

    keypoints: list[dict] = []
    for mp_idx in range(NUM_MP_KEYPOINTS):
        if mp_idx in raw:
            relative = zed_world_to_unity_absolute(raw[mp_idx] - hip_world)
            confidence = conf[mp_idx]
            source = "zed"
        else:
            relative = np.zeros(3, dtype=np.float64)
            confidence = 0.0
            source = "none"

        keypoints.append({
            "id": mp_idx,
            "name": MP_NAMES.get(mp_idx, f"MP_{mp_idx}"),
            "x": float(relative[0]),
            "y": float(relative[1]),
            "z": float(relative[2]),
            "confidence": confidence,
            "source": source,
        })

    return keypoints


def draw_skeleton_2d(
    image: np.ndarray,
    body: Any,
    image_scale: float = 1.0,
) -> None:
    """在 BGR 图像上绘制 ZED BODY_34 全骨架。"""
    import cv2

    if not hasattr(body, "keypoint_2d"):
        return

    points: dict[int, tuple[int, int]] = {}
    for zed_idx in range(NUM_ZED_KEYPOINTS):
        kp = body.keypoint_2d[zed_idx] * image_scale
        if kp[0] < 0 or kp[1] < 0:
            continue
        points[zed_idx] = (int(kp[0]), int(kp[1]))
        cv2.circle(image, points[zed_idx], 3, (0, 255, 0), -1)

    for a, b in BODY_34_BONES:
        if a in points and b in points:
            cv2.line(image, points[a], points[b], (255, 0, 0), 2)
