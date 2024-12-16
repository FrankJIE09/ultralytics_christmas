import os
import numpy as np


# 转换为YOLO_OBB格式的函数
def convert_to_yolo_obb(yolo_data, img_width, img_height):
    obb_data = ["YOLO_OBB"]  # 添加文件的第一行
    for line in yolo_data:
        try:
            if 'YOLO_OBB' in line or len(line.strip()) == 0:
                continue

            # 将每行的数据分割为浮点数，并确保至少有9个值（class_id + 8个坐标）
            items = list(map(float, line.split()))
            if len(items) != 9:
                print(f"Skipping malformed line: {line}")
                continue

            class_id = int(items[0])

            # 提取相对坐标的四个顶点
            x1_rel, y1_rel = items[1], items[2]
            x2_rel, y2_rel = items[3], items[4]
            x3_rel, y3_rel = items[5], items[6]
            x4_rel, y4_rel = items[7], items[8]

            # 将相对坐标转换为像素坐标
            x1, y1 = x1_rel * img_width, y1_rel * img_height
            x2, y2 = x2_rel * img_width, y2_rel * img_height
            x3, y3 = x3_rel * img_width, y3_rel * img_height
            x4, y4 = x4_rel * img_width, y4_rel * img_height

            # 计算中心点 (四个顶点的平均值)
            cx = (x1 + x2 + x3 + x4) / 4.0
            cy = (y1 + y2 + y3 + y4) / 4.0

            # 计算宽度和高度 (使用对边的距离)
            width = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            height = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)

            # 计算旋转角度 (与水平线的夹角)
            angle = -np.arctan2(y2 - y1, x2 - x1) * (180.0 / np.pi)  # 转化为角度

            # 保存为 YOLO_OBB 格式
            obb_data.append(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f} {angle:.6f}")

        except Exception as e:
            print(f"Error processing line: {line}, error: {e}")
            continue

    return obb_data


# 处理文件夹中的所有txt文件
def process_txt_files(input_folder, img_width, img_height):
    # 遍历文件夹中的所有txt文件
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)

            try:
                # 读取文件内容
                with open(file_path, 'r') as file:
                    yolo_txt_data = file.readlines()

                # 去除每行的换行符
                yolo_txt_data = [line.strip() for line in yolo_txt_data]

                # 转换为YOLO_OBB格式
                yolo_obb_data = convert_to_yolo_obb(yolo_txt_data, img_width, img_height)

                # 保存转换后的文件 (覆盖原文件)
                with open(file_path, 'w') as file:
                    for line in yolo_obb_data:
                        file.write(line + '\n')

            except Exception as e:
                print(f"Error processing file: {filename}, error: {e}")
                continue


# 示例：指定输入文件夹、图像尺寸
input_folder = '../GraspErzi/detections/original_images/2024-09-18'  # 输入文件夹路径
img_width = 1280  # 假设图像宽度
img_height = 720  # 假设图像高度

# 处理输入文件夹中的所有txt文件
process_txt_files(input_folder, img_width, img_height)
