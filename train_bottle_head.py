"""
文件描述：
该文件定义了一个自定义数据集类 `CustomDataset`，用于加载图像和标签数据，并应用数据增强变换。文件还包含YOLO模型的训练和预测逻辑，
根据 `train` 标志决定是训练模型还是使用训练好的模型进行预测，并显示预测结果。

作者：JieYu
日期：2024-07-25
"""
from ultralytics import YOLO
import torch

import json
import os

# 获取 settings.json 文件路径
settings_path = os.path.expanduser("~/.config/Ultralytics/settings.json")

# 读取当前的 settings.json 文件
with open(settings_path, "r") as f:
    settings = json.load(f)
project_dir = os.path.dirname(os.path.abspath(__file__))
# 修改默认路径为项目路径
settings["datasets_dir"] = f"{project_dir}/christmas_dataset"
settings["weights_dir"] = f"{project_dir}/weights"
settings["runs_dir"] = f"{project_dir}/runs"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 验证更改是否生效
print("Datasets directory:", settings["datasets_dir"])
print("Weights directory:", settings["weights_dir"])
print("Runs directory:", settings["runs_dir"])

train = True

dictionary = '/home/frank/Frank/code/ultralytics_christmas/christmas_dataset/bottle_head_dataset/'
if train:
    # 初始化 yolov8m-obb 模型
    model = YOLO('yolov8m', task="train")  # 确保模型路径正确
    # 训练模型
    model.train(data=dictionary + 'data.yaml', epochs=30, batch=16, imgsz=640)  # 确保数据路径和参数正确

    # 进行预测以验证训练效果
    results = model.predict(source=dictionary+'images/test')

    # 显示结果
    for item in results:
        item.show()
else:
    # 使用训练好的 yolov8m-obb 模型进行预测
    model = YOLO('./runs/obb/train10/weights/best.pt', task="val",)  # 确保模型路径正确
    # 预测
    results = model.predict(source=dictionary+'images/test')

    # 显示结果
    for item in results:
        item.show(font_size=1, line_width=2)
