# ski-cv

RealSense D435 运动姿态捕捉：RGB 姿态估计 + 深度反投影为 3D 关键点。

## 硬件

- Intel RealSense D435（USB 3.0）
- [RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense)

## 环境

```powershell
conda create -n ski python=3.10 -y
conda activate ski
pip install -r requirements.txt
```

## 运行

MediaPipe（33 关键点）：

```powershell
python realsense_mediapipe.py
```

YOLO11n-Pose（17 关键点）：

```powershell
python realsense_yolo.py
```

首次运行 MediaPipe 时会自动下载 `pose_landmarker_full.task` 到 `%LOCALAPPDATA%\cv_pose_models\`。

## 文件说明

| 文件 | 说明 |
|------|------|
| `realsense_mediapipe.py` | RealSense + MediaPipe 全身姿态 |
| `realsense_yolo.py` | RealSense + YOLO 姿态 + 骨长约束 |
| `joint_angles.py` | 关节角计算与正运动学可视化 |
| `yolo11n-pose.pt` | YOLO 预训练权重 |
