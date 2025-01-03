import os
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw

# 定义保存 ROI 的文件夹
image_dir = './collect_dataset/belt_images'
os.makedirs(image_dir, exist_ok=True)

# 使用训练好的 yolov8m-obb 模型进行预测
model = YOLO('./runs/obb/train3/weights/best.pt', task="val")  # 确保模型路径正确

# 预测
results = model.predict(source='./collect_dataset/images')

# 保存每个 OBB 单独的 ROI
for i, item in enumerate(results):
    img_path = item.path  # 获取原图路径
    img = Image.open(img_path).convert("RGB")
    img_width, img_height = img.size  # 获取图像尺寸

    if item.obb and len(item.obb) > 0:  # 确保 OBB 检测结果存在
        for j, obb_group in enumerate(item.obb):  # 遍历每组 OBB 检测
            for k, obb in enumerate(obb_group.xyxyxyxy):  # 遍历每个 OBB
                # 创建掩码图像
                masked_img = Image.new("L", (img_width, img_height), 0)
                draw = ImageDraw.Draw(masked_img)

                # 将 Tensor 转换为 NumPy 数组并转换为整数
                points = np.array(obb.tolist(), dtype=np.int32).reshape(-1, 2)

                # 计算扩展边界
                x_min = max(0, points[:, 0].min() - 50)
                y_min = max(0, points[:, 1].min() - 50)
                x_max = min(img_width, points[:, 0].max() + 50)
                y_max = min(img_height, points[:, 1].max() + 50)

                # 在掩码图像上绘制检测区域
                draw.rectangle([x_min, y_min, x_max, y_max], fill=255)

                # 将掩码区域合并到原始图像上
                roi_img = Image.composite(img, Image.new("RGB", (img_width, img_height), (0, 0, 0)), masked_img)

                # 保存单独的 ROI 图像
                save_path = os.path.join(image_dir, f"roi_{i}_{j}_{k}.png")
                roi_img.save(save_path)
