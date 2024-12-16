import os  # 导入操作系统相关的模块
import cv2  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入NumPy库，用于数值计算


def yolo_to_pixel_coords(image, yolo_coords):
    # 将YOLO格式的归一化坐标转换为图像中的像素坐标
    h, w = image.shape[:2]  # 获取图像的高度和宽度
    # 根据图像的高度和宽度，将YOLO归一化坐标转换为像素坐标
    coords = [(int(x * w), int(y * h)) for x, y in yolo_coords]
    return coords  # 返回转换后的像素坐标


def extract_and_save_objects(image_file, yolo_file, output_dir):
    # 从图像文件和YOLO标签文件中提取对象，并将其保存到指定目录
    image = cv2.imread(image_file)  # 读取图像文件
    h, w = image.shape[:2]  # 获取图像的高度和宽度

    # 打开YOLO标签文件并读取内容
    with open(yolo_file, 'r') as file:
        for line_num, line in enumerate(file):
            parts = line.strip().split()  # 将每一行分割成多个部分
            class_id = parts[0]  # 获取类别ID
            coords = list(map(float, parts[1:]))  # 将坐标部分转换为浮点数列表

            # 将YOLO格式的坐标成对读取并转换为像素坐标
            points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
            pixel_coords = yolo_to_pixel_coords(image, points)

            # 创建掩膜
            mask = np.zeros((h, w), dtype=np.uint8)  # 创建一个与图像大小相同的全黑掩膜
            pts = np.array(pixel_coords, np.int32)  # 将像素坐标转换为NumPy数组
            pts = pts.reshape((-1, 1, 2))  # 重塑数组为OpenCV填充多边形所需的形状
            cv2.fillPoly(mask, [pts], (255))  # 用白色填充掩膜上的多边形区域

            # 应用掩膜
            masked_image = cv2.bitwise_and(image, image, mask=mask)  # 将掩膜应用到图像上

            # 将其他区域设为黑色
            background = np.zeros_like(image)  # 创建一个与图像大小相同的全黑背景
            result_image = np.where(masked_image, masked_image, background)  # 合成结果图像

            # 创建类别目录
            class_dir = os.path.join(output_dir, class_id)
            os.makedirs(class_dir, exist_ok=True)  # 如果目录不存在，则创建

            # 生成输出文件路径
            output_file = os.path.join(class_dir, f"{os.path.basename(image_file).split('.')[0]}_{line_num}.jpg")
            cv2.imwrite(output_file, result_image)  # 保存结果图像到输出文件


def process_all_images(image_dir, label_dir, output_dir):
    # 处理所有图像和标签文件
    for filename in os.listdir(label_dir):  # 遍历标签目录中的所有文件
        if filename.endswith(".txt"):  # 只处理以.txt结尾的文件
            yolo_file = os.path.join(label_dir, filename)  # 构建YOLO标签文件的完整路径
            image_file = os.path.join(image_dir, filename.replace(".txt", ".jpg"))  # 构建对应的图像文件路径
            if os.path.exists(image_file):  # 如果图像文件存在
                extract_and_save_objects(image_file, yolo_file, output_dir)  # 提取并保存对象
            else:
                image_file = os.path.join(image_dir, filename.replace(".txt", ".png"))  # 构建对应的图像文件路径
                if os.path.exists(image_file):  # 如果图像文件存在
                    extract_and_save_objects(image_file, yolo_file, output_dir)  # 提取并保存对象


# 使用示例
image_dir = '../GraspErzi/detections/original_images/2024-09-18-2/filter/images/train'  # 图像文件所在目录
label_dir = '../GraspErzi/detections/original_images/2024-09-18-2/filter/labels/train'  # YOLO标签文件所在目录
output_dir = '../GraspErzi/detections/original_images/mask'  # 输出目录
process_all_images(image_dir, label_dir, output_dir)  # 处理所有图像和标签文件
