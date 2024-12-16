"""
文件描述：
该文件定义了一个自定义数据集类 `CustomDataset`，用于加载图像和标签数据，并应用数据增强变换。文件还包含YOLO模型的训练和预测逻辑，
根据 `train` 标志决定是训练模型还是使用训练好的模型进行预测，并显示预测结果。

作者：JieYu & Zhang Yiheng
日期：2024-07-25
"""
from ultralytics import YOLO
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import cv2
import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train = True

dictionary = '/home/frank/Frank/code/yolov10/GraspErzi/detections/original_images/2024-09-18-2/filter/'
if train is True:
    # 初始化模型
    model = YOLO('./runs/obb/cuvette_best2.pt', task="obb").to(device)
    # model =YOLO('./runs/obb/train14/weights/best.pt', task="obb")
    # 训练模型
    model.train(data=dictionary+'data.yaml', epochs=20, batch=64, imgsz=640)

    results = model.predict(source='./GraspErzi/detections/original_images/2024-09-18-2/test')

    # 显示结果
    for item in results:
        item.show()
else:

    model =YOLO('./runs/obb/testtube_v0919.pt', task="obb").to(device)
    # 预测
    results = model.predict(source='./GraspErzi/detections/original_images/2024-09-18-2/test')

    # 显示结果
    for item in results:
        item.show(font_size=1,line_width=2)
        
