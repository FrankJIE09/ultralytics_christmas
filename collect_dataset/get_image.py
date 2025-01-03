import cv2
import pyrealsense2 as rs
import os
import re
import numpy as np
# 确保保存图片的文件夹存在
image_folder = "images"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

# 初始化RealSense相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

# 检查文件夹中已有的图片编号，并找到最大的编号
max_num = 0
for filename in os.listdir(image_folder):
    if filename.endswith(".png"):
        num = int(re.findall(r'\d+', filename)[0])
        if num > max_num:
            max_num = num
image_count = max_num  # 开始于最大编号

print("按下 'r' 键保存图片，按下 'q' 键退出。")

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    # 获取彩色图像数据
    color_image = np.asanyarray(color_frame.get_data())
    cv2.imshow('RealSense', color_image)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        # 保存图片
        image_count += 1
        image_path = os.path.join(image_folder, f"{image_count}.png")
        cv2.imwrite(image_path, color_image)
        print(f"图片 {image_count} 已保存到 {image_path}")
    elif key == ord('q'):
        print("退出程序。")
        break

# 停止相机管道
pipeline.stop()
cv2.destroyAllWindows()
