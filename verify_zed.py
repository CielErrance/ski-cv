"""ZED 2i 硬件验证脚本（无 GUI）。"""

from __future__ import annotations

import sys
import time

import pyzed.sl as sl

from joint_angles import JOINT_ANGLE_KEYS, JointAngleCalculator
from realsense_mediapipe import build_broadcast_payload
from zed_skeleton import zed_keypoints_to_unity


def main() -> int:
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

    status = zed.open(init)
    print("open:", status)
    if status != sl.ERROR_CODE.SUCCESS:
        return 1

    info = zed.get_camera_information()
    print("device:", info.camera_model, "SN:", info.serial_number)

    tp = sl.PositionalTrackingParameters()
    tp.set_floor_as_origin = True
    print("positional tracking:", zed.enable_positional_tracking(tp))

    bp = sl.BodyTrackingParameters()
    bp.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
    bp.body_format = sl.BODY_FORMAT.BODY_34
    bp.enable_tracking = True
    bp.enable_body_fitting = True
    print("body tracking:", zed.enable_body_tracking(bp))

    bodies = sl.Bodies()
    brt = sl.BodyTrackingRuntimeParameters()
    brt.detection_confidence_threshold = 40
    calc = JointAngleCalculator()

    print("warming up 30 frames...")
    for _ in range(30):
        zed.grab()

    found = False
    max_conf = 0.0
    for i in range(120):
        if zed.grab() != sl.ERROR_CODE.SUCCESS:
            continue
        zed.retrieve_bodies(bodies, brt)
        if bodies.body_list:
            max_conf = max(max_conf, bodies.body_list[0].confidence)
            body = bodies.body_list[0]
            kps = zed_keypoints_to_unity(body)
            if kps:
                angles = calc.compute(kps)
                payload = build_broadcast_payload({
                    "timestamp": time.time(),
                    "joint_angles": angles,
                })
                assert "Time" in payload
                assert len(angles) == len(JOINT_ANGLE_KEYS)
                hip = angles.get("Hip_rAX", 0.0)
                print(
                    f"frame {i + 1}: conf={body.confidence:.1f}, "
                    f"angles={len(angles)}, Hip_rAX={hip:.1f}"
                )
                found = True
                break

    empty = build_broadcast_payload({"timestamp": time.time(), "joint_angles": {}})
    assert list(empty.keys()) == ["Time"]
    print("empty frame payload OK")

    zed.disable_body_tracking()
    zed.disable_positional_tracking()
    zed.close()

    if found:
        print("FULL PIPELINE OK")
        return 0

    print(f"NO BODY in 120 frames (max_conf={max_conf:.1f}) — 请站在相机前重试")
    return 2


if __name__ == "__main__":
    sys.exit(main())
