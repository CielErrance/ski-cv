"""关节角 ↔ FK 重建误差检验（合成特殊姿态，无需相机）。

用法（请先 conda activate ski）:
    python validate_joint_angles.py              # 固定用例
    python validate_joint_angles.py --random 500 # 随机合法姿态（FK 生成）
    python validate_joint_angles.py --random 500 --seed 42
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass

import numpy as np

from joint_angles import (
    JOINT_ANGLE_KEYS,
    MP,
    ForwardKinematics,
    JointAngleCalculator,
)

# FK 节点名 → MediaPipe 索引
NODE_TO_MP: dict[str, int] = {
    "hip_l": MP["LH"],
    "hip_r": MP["RH"],
    "knee_l": MP["LK"],
    "knee_r": MP["RK"],
    "ankle_l": MP["LA"],
    "ankle_r": MP["RA"],
    "foot_l": MP["LFI"],
    "foot_r": MP["RFI"],
    "shoulder_l": MP["LS"],
    "shoulder_r": MP["RS"],
    "elbow_l": MP["LE"],
    "elbow_r": MP["RE"],
    "wrist_l": MP["LW"],
    "wrist_r": MP["RW"],
    "hand_l": MP["LI"],
    "hand_r": MP["RI"],
}

# 锁 twist / 无脚板标记的角：不参与 round-trip 严格比较
LOCKED_ANGLE_KEYS = {
    "Elbow_lAZ", "Elbow_rAZ",
    "Ankle_lAY", "Ankle_rAY",
    "Wrist_lAY", "Wrist_rAY",
}

POS_TOL_M = 0.02          # 单点位置误差上限（m）
RMSE_TOL_M = 0.012        # 12 点 RMSE 上限（m）
ANGLE_TOL_RAD = 0.08      # 角度 round-trip 上限（rad，≈4.6°）
LOCKED_TOL_RAD = 1e-6     # 锁定角应接近 0

# 随机姿态：各角采样范围（度），锁定角固定为 0
RANDOM_ANGLE_RANGES_DEG: dict[str, tuple[float, float]] = {
    "Pelvis_AX": (-15.0, 15.0),
    "Pelvis_AY": (-15.0, 15.0),
    "Pelvis_AZ": (-20.0, 20.0),
    "Hip_rAX": (-20.0, 70.0),
    "Hip_rAY": (-25.0, 25.0),
    "Hip_rAZ": (-30.0, 30.0),
    "Hip_lAX": (-20.0, 70.0),
    "Hip_lAY": (-25.0, 25.0),
    "Hip_lAZ": (-30.0, 30.0),
    "Knee_rAX": (-120.0, 5.0),
    "Knee_rAY": (-15.0, 15.0),
    "Knee_rAZ": (-15.0, 15.0),
    "Knee_lAX": (-120.0, 5.0),
    "Knee_lAY": (-15.0, 15.0),
    "Knee_lAZ": (-15.0, 15.0),
    "Ankle_rAX": (-25.0, 25.0),
    "Ankle_rAZ": (-25.0, 25.0),
    "Ankle_lAX": (-25.0, 25.0),
    "Ankle_lAZ": (-25.0, 25.0),
    "Arm_rAX": (-60.0, 60.0),
    "Arm_rAY": (-45.0, 45.0),
    "Arm_rAZ": (-30.0, 100.0),
    "Arm_lAX": (-60.0, 60.0),
    "Arm_lAY": (-45.0, 45.0),
    "Arm_lAZ": (-30.0, 100.0),
    "Elbow_rAX": (0.0, 150.0),
    "Elbow_rAY": (-90.0, 90.0),
    "Elbow_lAX": (0.0, 150.0),
    "Elbow_lAY": (-90.0, 90.0),
    "Wrist_rAX": (-40.0, 40.0),
    "Wrist_lAX": (-40.0, 40.0),
}
for _k in LOCKED_ANGLE_KEYS:
    RANDOM_ANGLE_RANGES_DEG[_k] = (0.0, 0.0)
for _k in JOINT_ANGLE_KEYS:
    RANDOM_ANGLE_RANGES_DEG.setdefault(_k, (0.0, 0.0))


@dataclass
class PoseCase:
    name: str
    keypoints: list[dict]
    note: str = ""
    required: bool = True  # False = 应力用例，仅报告不判失败
    source_angles: dict[str, float] | None = None  # 随机 FK 合成时的原始关节角


def _kp(idx: int, x: float, y: float, z: float, *, confidence: float = 1.0) -> dict:
    return {
        "id": idx,
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "confidence": float(confidence),
        "source": "synthetic",
    }


def _blank_keypoints() -> list[dict]:
    return [_kp(i, 0.0, 0.0, 0.0, confidence=0.0) for i in range(33)]


def _set_point(kps: list[dict], idx: int, p: np.ndarray) -> None:
    for kp in kps:
        if kp["id"] == idx:
            kp["x"], kp["y"], kp["z"] = float(p[0]), float(p[1]), float(p[2])
            kp["confidence"] = 1.0
            return
    kps.append(_kp(idx, float(p[0]), float(p[1]), float(p[2])))


def _keypoint_present(kps: list[dict], idx: int) -> bool:
    for kp in kps:
        if kp["id"] != idx:
            continue
        if float(kp.get("confidence", 0.0)) < 0.5:
            return False
        p = np.array([kp["x"], kp["y"], kp["z"]], dtype=np.float64)
        return bool(np.all(np.isfinite(p)) and np.linalg.norm(p) > 1e-5)
    return False


# 仅用于 FK 可视化补全的远端点；round-trip 时若源数据无该点则忽略
DISTAL_MP_FOR_ROUNDTRIP = frozenset({
    MP["LI"], MP["RI"], MP["LFI"], MP["RFI"],
})


def _clear_absent_distals(kps: list[dict], source: list[dict]) -> None:
    for kp in kps:
        if kp["id"] not in DISTAL_MP_FOR_ROUNDTRIP:
            continue
        if _keypoint_present(source, kp["id"]):
            continue
        kp["x"] = kp["y"] = kp["z"] = 0.0
        kp["confidence"] = 0.0


def _points_from_nodes(nodes: dict[str, np.ndarray]) -> list[dict]:
    kps = _blank_keypoints()
    for node, mp_id in NODE_TO_MP.items():
        if node in nodes:
            _set_point(kps, mp_id, nodes[node])
    return kps


def _keypoint_vec(kps: list[dict], idx: int) -> np.ndarray:
    for kp in kps:
        if kp["id"] == idx:
            return np.array([kp["x"], kp["y"], kp["z"]], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)


@dataclass
class PoseReport:
    name: str
    pos_errors_m: dict[str, float]
    rmse_m: float
    max_pos_m: float
    angle_errors_rad: dict[str, float]
    max_angle_rad: float
    locked_ok: bool
    passed: bool
    required: bool
    note: str = ""


def _build_fk_pose(
    name: str,
    angles: dict[str, float],
    anchors: dict[str, np.ndarray],
    lengths: dict[str, float],
    note: str = "FK 合成（角度→节点，作 ground truth）",
) -> PoseCase:
    fk = ForwardKinematics()
    fk.anchor_offsets = {k: v.copy() for k, v in anchors.items()}
    fk.segment_lengths = dict(lengths)
    nodes = fk.rebuild(angles)
    return PoseCase(
        name=name,
        keypoints=_points_from_nodes(nodes),
        note=note,
        source_angles=dict(angles),
    )


def _standard_proportions() -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """与 T 姿一致的标准骨长 / 锚点。"""
    t_pose = _t_pose_keypoints()
    fk = ForwardKinematics()
    fk.update_lengths(t_pose)
    return dict(fk.anchor_offsets), dict(fk.segment_lengths)


def _t_pose_keypoints() -> list[dict]:
    """直立 T 姿：双臂侧平举，双腿竖直。"""
    kps = _blank_keypoints()
    pelvis = np.array([0.0, 0.0, 0.0])
    half_w = 0.10
    shoulder_y = 0.35
    elbow_y = 0.35
    wrist_y = 0.35
    knee_y = -0.40
    ankle_y = -0.80
    ankle_z = 0.05

    for idx, x in ((MP["LH"], -half_w), (MP["RH"], half_w)):
        _set_point(kps, idx, pelvis + np.array([x, 0.0, 0.0]))
    for idx, x in ((MP["LS"], -half_w), (MP["RS"], half_w)):
        _set_point(kps, idx, pelvis + np.array([x, shoulder_y, 0.0]))
    for idx, x in ((MP["LE"], -0.35), (MP["RE"], 0.35)):
        _set_point(kps, idx, np.array([x, elbow_y, 0.0]))
    for idx, x in ((MP["LW"], -0.55), (MP["RW"], 0.55)):
        _set_point(kps, idx, np.array([x, wrist_y, 0.0]))
    for idx, x in ((MP["LK"], -half_w), (MP["RK"], half_w)):
        _set_point(kps, idx, np.array([x, knee_y, 0.0]))
    for idx, x in ((MP["LA"], -half_w), (MP["RA"], half_w)):
        _set_point(kps, idx, np.array([x, ankle_y, ankle_z]))
    return kps


def _squat_keypoints() -> list[dict]:
    """深蹲：大腿前倾、小腿近竖直、膝大幅弯曲。"""
    kps = _t_pose_keypoints()
    half_w = 0.10
    # 右膝前移且下沉
    _set_point(kps, MP["RK"], np.array([half_w + 0.08, -0.22, 0.18]))
    _set_point(kps, MP["RA"], np.array([half_w + 0.05, -0.72, 0.10]))
    _set_point(kps, MP["LK"], np.array([-half_w - 0.08, -0.22, 0.18]))
    _set_point(kps, MP["LA"], np.array([-half_w - 0.05, -0.72, 0.10]))
    return kps


def _right_leg_forward_keypoints() -> list[dict]:
    """右髋屈曲：右大腿前抬约 45°。"""
    kps = _t_pose_keypoints()
    half_w = 0.10
    thigh_len = 0.40
    angle = math.radians(45.0)
    hip = np.array([half_w, 0.0, 0.0])
    knee = hip + np.array([0.0, math.cos(angle), math.sin(angle)]) * thigh_len
    ankle = knee + np.array([0.0, -0.40, 0.05])
    _set_point(kps, MP["RK"], knee)
    _set_point(kps, MP["RA"], ankle)
    _set_point(kps, MP["RFI"], ankle + np.array([0.0, -0.05, 0.18]))
    return kps


def _right_elbow_flex_keypoints() -> list[dict]:
    """右肘屈曲约 90°，前臂向前。"""
    kps = _t_pose_keypoints()
    half_w = 0.10
    shoulder = np.array([half_w, 0.35, 0.0])
    elbow = np.array([0.35, 0.35, 0.0])
    wrist = np.array([0.35, 0.10, 0.20])
    _set_point(kps, MP["RE"], elbow)
    _set_point(kps, MP["RW"], wrist)
    _set_point(kps, MP["RI"], wrist + np.array([0.0, -0.05, 0.12]))
    # 保持左臂 T 姿
    return kps


def _arm_forward_keypoints() -> list[dict]:
    """双臂前平举（沿 +Z）。"""
    kps = _t_pose_keypoints()
    half_w = 0.10
    for shoulder_x, elbow_id, wrist_id, sign in (
        (-half_w, MP["LE"], MP["LW"], -1),
        (half_w, MP["RE"], MP["RW"], 1),
    ):
        s = np.array([shoulder_x, 0.35, 0.0])
        e = s + np.array([0.0, 0.0, 0.25])
        w = e + np.array([0.0, 0.0, 0.25])
        _set_point(kps, elbow_id, e)
        _set_point(kps, wrist_id, w)
    return kps


def _pelvis_tilt_keypoints() -> list[dict]:
    """骨盆前倾：肩髋连线仍建系，整体微倾。"""
    kps = _t_pose_keypoints()
    tilt = math.radians(12.0)
    c, s = math.cos(tilt), math.sin(tilt)
    rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)
    origin = np.zeros(3)
    for kp in kps:
        if kp["id"] in set(NODE_TO_MP.values()):
            p = np.array([kp["x"], kp["y"], kp["z"]])
            q = origin + rot @ (p - origin)
            kp["x"], kp["y"], kp["z"] = float(q[0]), float(q[1]), float(q[2])
    return kps


def _fk_synthetic_poses() -> list[PoseCase]:
    """由已知关节角 FK 生成节点，检验 encode/decode 自洽性。"""
    anchors, lengths = _standard_proportions()
    cases: list[PoseCase] = []

    base = {k: 0.0 for k in JOINT_ANGLE_KEYS}

    standing = dict(base)
    cases.append(_build_fk_pose("fk_standing", standing, anchors, lengths))

    hip_flex = dict(base)
    hip_flex["Hip_rAX"] = math.radians(40.0)
    cases.append(_build_fk_pose("fk_hip_flex_r", hip_flex, anchors, lengths))

    knee_flex = dict(base)
    knee_flex["Knee_rAX"] = math.radians(-60.0)
    cases.append(_build_fk_pose("fk_knee_flex_r", knee_flex, anchors, lengths))

    elbow_flex = dict(base)
    elbow_flex["Elbow_rAX"] = math.radians(90.0)
    cases.append(_build_fk_pose("fk_elbow_flex_r", elbow_flex, anchors, lengths))

    pelvis = dict(base)
    pelvis["Pelvis_AX"] = math.radians(-8.0)
    pelvis["Pelvis_AZ"] = math.radians(15.0)
    cases.append(_build_fk_pose("fk_pelvis_rot", pelvis, anchors, lengths))

    combo = dict(base)
    combo["Hip_lAX"] = math.radians(30.0)
    combo["Knee_lAX"] = math.radians(-45.0)
    combo["Arm_rAZ"] = math.radians(35.0)
    combo["Elbow_rAX"] = math.radians(70.0)
    cases.append(_build_fk_pose("fk_combo", combo, anchors, lengths))

    arm_fwd = dict(base)
    arm_fwd["Arm_lAZ"] = math.radians(90.0)
    arm_fwd["Arm_rAZ"] = math.radians(90.0)
    cases.append(_build_fk_pose("fk_arms_forward", arm_fwd, anchors, lengths))

    return cases


def _stress_pose_cases() -> list[PoseCase]:
    """手动关键点，可能超出当前 encode/decode 约定；仅作诊断。"""
    return [
        PoseCase(
            "stress_right_elbow_flex",
            _right_elbow_flex_keypoints(),
            "右肘 ~90° 前臂前伸",
            required=False,
        ),
        PoseCase(
            "stress_arms_forward",
            _arm_forward_keypoints(),
            "双臂前平举",
            required=False,
        ),
    ]


def all_pose_cases() -> list[PoseCase]:
    manual = [
        PoseCase("t_pose", _t_pose_keypoints(), "直立 T 姿"),
        PoseCase("squat", _squat_keypoints(), "深蹲"),
        PoseCase("right_leg_forward", _right_leg_forward_keypoints(), "右髋屈曲"),
        PoseCase("pelvis_tilt", _pelvis_tilt_keypoints(), "骨盆前倾 ~12°"),
    ]
    return manual + _fk_synthetic_poses() + _stress_pose_cases()


def _random_angles(rng: np.random.Generator) -> dict[str, float]:
    """在约定范围内均匀采样一组关节角（弧度）。"""
    angles: dict[str, float] = {}
    for key in JOINT_ANGLE_KEYS:
        lo, hi = RANDOM_ANGLE_RANGES_DEG[key]
        deg = float(rng.uniform(lo, hi)) if lo != hi else lo
        angles[key] = math.radians(deg)
    return angles


def random_pose_cases(
    count: int,
    *,
    seed: int | None = None,
) -> list[PoseCase]:
    """随机关节角 → FK 节点 → 合法 3D 关键点（12 点）。"""
    rng = np.random.default_rng(seed)
    anchors, lengths = _standard_proportions()
    cases: list[PoseCase] = []
    for i in range(count):
        angles = _random_angles(rng)
        cases.append(_build_fk_pose(
            f"random_{i:04d}",
            angles,
            anchors,
            lengths,
            note="随机关节角 FK 合成",
        ))
    return cases


@dataclass
class RandomSummary:
    count: int
    passed_recon: int
    pos_rmse_mm: np.ndarray
    pos_max_mm: np.ndarray
    angle_max_deg: np.ndarray
    encode_drift_deg: np.ndarray
    locked_leak_deg: np.ndarray
    worst_case: str
    worst_rmse_mm: float


def evaluate_random_poses(
    count: int,
    *,
    seed: int | None = None,
    verbose_failures: int = 5,
) -> RandomSummary:
    """批量评估随机合法姿态，返回汇总统计。"""
    cases = random_pose_cases(count, seed=seed)
    reports = [evaluate_pose(c, check_locked=False) for c in cases]

    rmse = np.array([r.rmse_m * 1000 for r in reports], dtype=np.float64)
    max_pos = np.array([r.max_pos_m * 1000 for r in reports], dtype=np.float64)
    max_ang = np.array([math.degrees(r.max_angle_rad) for r in reports], dtype=np.float64)
    passed_recon = sum(1 for r in reports if r.passed)

    encode_drifts: list[float] = []
    locked_leaks: list[float] = []
    calc = JointAngleCalculator()
    for case in cases:
        encoded = calc.compute(case.keypoints)
        if case.source_angles:
            drifts = [
                abs(math.degrees(float(encoded[k] - case.source_angles[k])))
                for k in JOINT_ANGLE_KEYS
                if k not in LOCKED_ANGLE_KEYS
            ]
            encode_drifts.append(max(drifts) if drifts else 0.0)
        leaks = [
            abs(math.degrees(float(encoded[k])))
            for k in LOCKED_ANGLE_KEYS
        ]
        locked_leaks.append(max(leaks) if leaks else 0.0)

    drift_arr = np.array(encode_drifts, dtype=np.float64)
    leak_arr = np.array(locked_leaks, dtype=np.float64)

    worst_idx = int(np.argmax(rmse))
    worst = reports[worst_idx]

    print(f"\n=== 随机合法姿态 FK 重建 ({count} 组, seed={seed}) ===")
    print("合法姿态定义: 随机关节角 → FK 生成 12 点 → compute → FK 重建")
    print(f"几何重建通过: {passed_recon}/{count} ({100.0 * passed_recon / count:.1f}%)")
    print(f"位置 RMSE (mm): mean={rmse.mean():.3f}  p95={np.percentile(rmse, 95):.3f}  max={rmse.max():.3f}")
    print(f"位置最大 (mm): mean={max_pos.mean():.3f}  p95={np.percentile(max_pos, 95):.3f}  max={max_pos.max():.3f}")
    print(f"角度 round-trip (°): mean={max_ang.mean():.4f}  p95={np.percentile(max_ang, 95):.4f}  max={max_ang.max():.4f}")
    print(f"编码漂移 |encode-source| (deg): mean={drift_arr.mean():.2f}  p95={np.percentile(drift_arr, 95):.2f}  max={drift_arr.max():.2f}")
    print(f"锁定角泄漏 max|locked| (deg): mean={leak_arr.mean():.2f}  p95={np.percentile(leak_arr, 95):.2f}  max={leak_arr.max():.2f}")

    failures = [(case, r) for case, r in zip(cases, reports) if not r.passed]
    if failures:
        print(f"\n几何重建失败 {len(failures)} 组，展示前 {min(verbose_failures, len(failures))} 组:")
        for _, r in failures[:verbose_failures]:
            _print_report(r)
    else:
        print("\n几何重建全部通过阈值（位置 + 非锁定角 round-trip）。")

    print(f"\n最差样本: {worst.name} — RMSE {worst.rmse_m * 1000:.3f} mm, "
          f"最大 {worst.max_pos_m * 1000:.3f} mm, 角度 {math.degrees(worst.max_angle_rad):.3f}°")

    return RandomSummary(
        count=count,
        passed_recon=passed_recon,
        pos_rmse_mm=rmse,
        pos_max_mm=max_pos,
        angle_max_deg=max_ang,
        encode_drift_deg=drift_arr,
        locked_leak_deg=leak_arr,
        worst_case=worst.name,
        worst_rmse_mm=float(rmse[worst_idx]),
    )


def evaluate_pose(case: PoseCase, *, check_locked: bool = True) -> PoseReport:
    calc = JointAngleCalculator()
    fk = ForwardKinematics()
    fk.update_lengths(case.keypoints)

    angles = calc.compute(case.keypoints)
    nodes = fk.rebuild(angles)

    pos_errors: dict[str, float] = {}
    sq_sum = 0.0
    n = 0
    for node, mp_id in NODE_TO_MP.items():
        if not _keypoint_present(case.keypoints, mp_id):
            continue
        src = _keypoint_vec(case.keypoints, mp_id)
        if node not in nodes:
            continue
        err = float(np.linalg.norm(src - nodes[node]))
        pos_errors[node] = err
        sq_sum += err * err
        n += 1
    rmse = math.sqrt(sq_sum / n) if n else 0.0
    max_pos = max(pos_errors.values()) if pos_errors else 0.0

    kps2 = _points_from_nodes(nodes)
    _clear_absent_distals(kps2, case.keypoints)
    angles2 = calc.compute(kps2)

    angle_errors: dict[str, float] = {}
    for key in JOINT_ANGLE_KEYS:
        if key in LOCKED_ANGLE_KEYS:
            continue
        angle_errors[key] = abs(float(angles[key] - angles2[key]))
    max_angle = max(angle_errors.values()) if angle_errors else 0.0

    locked_ok = (
        all(abs(float(angles[k])) < LOCKED_TOL_RAD for k in LOCKED_ANGLE_KEYS)
        if check_locked
        else True
    )

    passed = (
        max_pos <= POS_TOL_M
        and rmse <= RMSE_TOL_M
        and max_angle <= ANGLE_TOL_RAD
        and locked_ok
    )

    return PoseReport(
        name=case.name,
        pos_errors_m=pos_errors,
        rmse_m=rmse,
        max_pos_m=max_pos,
        angle_errors_rad=angle_errors,
        max_angle_rad=max_angle,
        locked_ok=locked_ok,
        passed=passed,
        required=case.required,
        note=case.note,
    )


def _print_report(report: PoseReport) -> None:
    if report.required:
        status = "PASS" if report.passed else "FAIL"
    else:
        status = "INFO" if report.passed else "WARN"
    print(f"\n[{status}] {report.name} — {report.note}")
    print(f"  位置 RMSE: {report.rmse_m * 1000:.1f} mm | 最大: {report.max_pos_m * 1000:.1f} mm")
    print(f"  角度 round-trip 最大: {math.degrees(report.max_angle_rad):.2f}°")
    if not report.locked_ok:
        print("  锁定角非零（应≈0）")

    worst_nodes = sorted(report.pos_errors_m.items(), key=lambda x: -x[1])[:3]
    if worst_nodes:
        parts = [f"{n}={e * 1000:.1f}mm" for n, e in worst_nodes]
        print(f"  最大误差点: {', '.join(parts)}")

    if report.max_angle_rad > ANGLE_TOL_RAD:
        worst_angles = sorted(report.angle_errors_rad.items(), key=lambda x: -x[1])[:5]
        parts = [f"{k}={math.degrees(v):.2f}°" for k, v in worst_angles]
        print(f"  最大角度差: {', '.join(parts)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="关节角 FK 重建检验")
    parser.add_argument(
        "--random", type=int, default=0, metavar="N",
        help="额外运行 N 组随机合法姿态（FK 生成 3D 点）",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（可复现）")
    args = parser.parse_args()

    print("关节角 FK 重建检验")
    print(f"阈值: 单点≤{POS_TOL_M * 1000:.0f}mm, RMSE≤{RMSE_TOL_M * 1000:.0f}mm, "
          f"角度≤{math.degrees(ANGLE_TOL_RAD):.1f}°")

    reports = [evaluate_pose(c) for c in all_pose_cases()]
    required = [r for r in reports if r.required]
    stress = [r for r in reports if not r.required]

    for r in required:
        _print_report(r)
    if stress:
        print("\n--- 应力用例（手动关键点） ---")
        for r in stress:
            _print_report(r)

    passed = sum(1 for r in required if r.passed)
    total = len(required)
    print(f"\n必过用例: {passed}/{total} 通过")

    exit_code = 0 if passed == total else 1

    if args.random > 0:
        summary = evaluate_random_poses(args.random, seed=args.seed)
        if summary.passed_recon < summary.count:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
