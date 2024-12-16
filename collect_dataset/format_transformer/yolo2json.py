import os
import json
import base64
from clearFile import clear_specific_files

# Directory path that contains YOLO txt files
# txt_path = '../GraspErzi/detections/original_images'
# pic_path = '../GraspErzi/detections/original_images'  # 请替换为你的文件夹路径
txt_path = '/home/frank/Frank/code/yolo11/datasets/valve_train/labels/val'
pic_path = '/home/frank/Frank/code/yolo11/datasets/valve_train/labels/val'  # 请替换为你的文件夹路径
file_extensions = ['.json']  # 指定要删除的文件类型
clear_specific_files(pic_path, file_extensions)

# Label names
labels = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15', '16']

# Number of points in the polygon (must be an even number)
poly_point = 10

# Image dimensions
img_width = 1280
img_height = 720

# Iterate through each file in the directory
for filename in os.listdir(txt_path):
    if filename.endswith(".txt"):
        print(f"Processing file: {filename}")

        # Read the contents of the txt file
        with open(os.path.join(txt_path, filename), 'r') as file:
            lines = file.readlines()

        shapes = []
        for line in lines:
            if 'YOLO_OBB' in line or len(line.strip()) == 0:
                continue
            content = line.strip().split()
            label_id = int(float(content[0]))
            xy = []

            # Read polygon points from the content
            for r in range(1, len(content), 2):
                x = float(content[r]) * img_width
                y = float(content[r + 1]) * img_height
                xy.append([x, y])

            # Create shape dictionary
            shape = {
                "label": labels[label_id - 1],
                "points": xy,
                "group_id": None,
                "shape_type": "polygon",
                "flags": {}
            }
            shapes.append(shape)

        # Try both jpg and png extensions
        image_filename_jpg = filename.replace(".txt", ".JPG")
        image_filename_png = filename.replace(".txt", ".png")

        image_data = None

        if os.path.exists(os.path.join(pic_path, image_filename_jpg)):
            image_filename = image_filename_jpg
            with open(os.path.join(pic_path, image_filename), "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
        elif os.path.exists(os.path.join(pic_path, image_filename_png)):
            image_filename = image_filename_png
            with open(os.path.join(pic_path, image_filename), "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")
        else:
            print(f"Image not found for {filename}")
            continue

        # Create JSON structure
        json_data = {
            "version": "5.2.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": image_filename,
            "imageData": image_data,
            "imageHeight": img_height,
            "imageWidth": img_width
        }

        # Write JSON data to file
        json_filename = filename.replace(".txt", ".json")
        with open(os.path.join(pic_path, json_filename), "w") as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"Saved JSON file: {json_filename}")
