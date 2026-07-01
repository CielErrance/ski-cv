# ski-cv

RealSense D435 / ZED 2i 运动姿态捕捉：RGB 姿态估计 + 3D 关键点 / 关节角输出。

## 硬件

- Intel RealSense D435（USB 3.0）+ [RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense)
- Stereolabs ZED 2i + [ZED SDK 4.x+](https://www.stereolabs.com/developers/release/)

## 环境

```powershell
conda create -n ski python=3.10 -y
conda activate ski
pip install -r requirements.txt
```

### ZED SDK / pyzed（仅 `zed_bodytracking.py` 需要）

`pyzed` 不能通过 `pip install pyzed` 安装，需先安装 ZED SDK，再用 SDK 自带的脚本生成 Python 包：

1. 从 [Stereolabs 发布页](https://www.stereolabs.com/developers/release/) 安装与相机匹配的 ZED SDK（ZED 2i 需 4.x+）
2. 在 SDK 安装目录运行 `get_python_api.py`，对生成的 `.whl` 执行 `pip install`
3. 验证：`python -c "import pyzed.sl; print('OK')"`
4. 运行前关闭 ZED Explorer 等占用相机的程序

## 运行

MediaPipe（RealSense D435，33 关键点）：

```powershell
python realsense_mediapipe.py
```

ZED Body Tracking（ZED 2i，BODY_34 → 关节角，输出格式与 MediaPipe 版相同）：

```powershell
python zed_bodytracking.py
```

录制文件保存在 `recordings/`，ZED 版文件名前缀为 `joint_angles_zed_*`。

YOLO11n-Pose（17 关键点）：

```powershell
python realsense_yolo.py
```

首次运行 MediaPipe 时会自动下载 `pose_landmarker_full.task` 到 `%LOCALAPPDATA%\cv_pose_models\`。

## 文件说明

| 文件 | 说明 |
|------|------|
| `realsense_mediapipe.py` | RealSense + MediaPipe 全身姿态 |
| `zed_bodytracking.py` | ZED 2i + Body Tracking 关节角采集 |
| `zed_skeleton.py` | ZED BODY_34 → MediaPipe 索引 / Unity 坐标适配 |
| `realsense_yolo.py` | RealSense + YOLO 姿态 + 骨长约束 |
| `joint_angles.py` | 关节角计算与正运动学可视化 |
| `yolo11n-pose.pt` | YOLO 预训练权重 |

## 发布到 GitHub

本地已配置远程 `https://github.com/CielErrance/ski-cv.git`。首次推送：

```powershell
gh auth login
.\publish.ps1
```

若 `github.com` HTTPS 超时，可开代理后重试，或将 `~/.ssh/id_rsa.pub` 添加到 GitHub SSH keys 后：

```powershell
$env:GIT_SSH_COMMAND = 'ssh -p 443 -o Hostname=ssh.github.com'
git remote set-url origin git@github.com:CielErrance/ski-cv.git
git push -u origin main
```
