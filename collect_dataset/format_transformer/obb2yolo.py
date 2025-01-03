import os
import numpy as np


# 从 YOLO_OBB 转换为 YOLO 格式的函数
def convert_to_yolo(yolo_obb_data, img_width, img_height):
    yolo_data = []
    for line in yolo_obb_data:
        try:
            # 跳过文件第一行 "YOLO_OBB"
            if 'YOLO_OBB' in line or len(line.strip()) == 0:
                print(f"Skipping header or empty line: {line}")
                continue

            # 将每行的数据分割为浮点数
            items = line.split()
            if len(items) == 6:
                # 处理YOLO_OBB格式
                items = list(map(float, items))
                class_id = int(items[0])
                cx, cy = items[1], items[2]
                width, height = items[3], items[4]
                angle = -items[5] * (np.pi / 180.0)  # 转换为弧度

                # 计算矩形的四个顶点
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)

                half_w = width / 2
                half_h = height / 2

                # 计算四个顶点相对于中心的偏移量
                x1 = cx + (half_w * cos_a - half_h * sin_a)
                y1 = cy + (half_w * sin_a + half_h * cos_a)
                x2 = cx + (-half_w * cos_a - half_h * sin_a)
                y2 = cy + (-half_w * sin_a + half_h * cos_a)
                x3 = cx + (-half_w * cos_a + half_h * sin_a)
                y3 = cy + (-half_w * sin_a - half_h * cos_a)
                x4 = cx + (half_w * cos_a + half_h * sin_a)
                y4 = cy + (half_w * sin_a - half_h * cos_a)

                # 转换为相对坐标
                x1_rel, y1_rel = x1 / img_width, y1 / img_height
                x2_rel, y2_rel = x2 / img_width, y2 / img_height
                x3_rel, y3_rel = x3 / img_width, y3 / img_height
                x4_rel, y4_rel = x4 / img_width, y4 / img_height

                # 保存为YOLO格式
                yolo_data.append(
                    f"{class_id} {x1_rel:.6f} {y1_rel:.6f} {x2_rel:.6f} {y2_rel:.6f} {x3_rel:.6f} {y3_rel:.6f} {x4_rel:.6f} {y4_rel:.6f}")
                print(f"Processed line: {line}")
            else:
                # 如果不是YOLO_OBB格式，则跳过（可能是已经是YOLO格式）
                print(f"Skipping malformed or non-YOLO_OBB line: {line}")
                continue

        except Exception as e:
            print(f"Error processing line: {line}, error: {e}")
            continue

    return yolo_data


# 处理文件夹中的所有txt文件
def process_obb_txt_files(input_folder, img_width, img_height):
    # 遍历文件夹中的所有txt文件
    for filename in os.listdir(input_folder):
        # 跳过 classes.txt 文件
        if filename == "classes.txt":
            continue

        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)

            try:
                # 读取文件内容
                with open(file_path, 'r') as file:
                    yolo_obb_data = file.readlines()

                # 去除每行的换行符
                yolo_obb_data = [line.strip() for line in yolo_obb_data]
                print(f"Processing file: {filename}, total lines: {len(yolo_obb_data)}")

                # 转换为YOLO格式
                yolo_data = convert_to_yolo(yolo_obb_data, img_width, img_height)

                # 检查生成的yolo_data是否有内容
                if len(yolo_data) > 0:
                    # 保存转换后的文件 (覆盖原文件)
                    with open(file_path, 'w') as file:
                        for line in yolo_data:
                            file.write(line + '\n')
                    print(f"File saved: {filename}")
                else:
                    print(f"No data to write for file: {filename}")

            except Exception as e:
                print(f"Error processing file: {filename}, error: {e}")
                continue


# 示例：指定输入文件夹、图像尺寸
input_folder = '../belt_images'  # 输入文件夹路径
img_width = 1280  # 假设图像宽度
img_height = 720  # 假设图像高度

# 处理输入文件夹中的所有txt文件
process_obb_txt_files(input_folder, img_width, img_height)
