# Fire Prop Detector（火焰道具检测）

基于 Ultralytics YOLOv8 的火焰道具检测模型，用于对火焰道具（fire prop）在相机 / 视频中的实时检测。提供 PyTorch 与 ONNX 两种部署权重，以及一个开箱即用的推理脚本。

## 检测类别

| 序号 | 类别 | 说明 |
|---|---|---|
| 0 | `fire_prop` | 火焰道具主体 |
| 1 | `black_base` | 黑色底座 |
| 2 | `red_cloth_strip` | 红色布条 |

## 文件说明

| 文件 | 说明 |
|---|---|
| `best.pt` | Ultralytics YOLOv8 训练得到的 PyTorch 权重 |
| `best.onnx` | 导出的 ONNX 模型，适合更快、更便携的部署（如树莓派） |
| `detect_pi.py` | 相机 / 视频推理脚本，实时画框显示 |

## 安装依赖

在树莓派等设备上推荐：

```bash
# 使用 PyTorch 权重
pip install ultralytics opencv-python

# 或使用 ONNX 部署
pip install onnxruntime opencv-python
```

## 使用

相机实时检测：

```bash
python detect_pi.py --model best.pt --source 0 --conf 0.45
```

检测视频文件并保存结果：

```bash
python detect_pi.py --model best.pt --source your_video.mp4 --conf 0.45 --save
```

## 参数

- `--model`：模型路径，默认 `best.pt`
- `--source`：`0` 表示相机，或视频 / 图片路径
- `--conf`：置信度阈值，默认 0.45
- `--imgsz`：推理分辨率，默认 640
- `--save`：是否保存输出视频（默认输出 `output.avi`）

## 说明

- 建议从 `conf=0.45` 开始，误检较多时可适当调高阈值。
- 在树莓派等算力有限的设备上，推荐使用 `best.onnx` + `onnxruntime`，推理更快、部署更便携。
- 相机窗口下按 `q` 退出。
