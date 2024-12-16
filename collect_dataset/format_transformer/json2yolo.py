import os
import json
from PIL import Image

# List of directory paths that contain labelme json files
paths = [
    "/home/frank/Frank/code/yolo11/datasets/valve_train/labels/val",
    "/home/frank/Frank/code/yolo11/datasets/valve_train/labels/train"
    # Add more paths if needed
]

# Iterate through each directory and file
for path in paths:
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            print(f"Processing file: {filename} in {path}")

            # Read the JSON file
            with open(os.path.join(path, filename), 'r') as file:
                data = json.load(file)

            # Get image file name from JSON and open the image to get dimensions
            image_path = os.path.join(path, data['imagePath'])
            with Image.open(image_path) as img:
                img_width, img_height = img.size

            txt_filename = filename.replace(".json", ".txt")
            with open(os.path.join(path, txt_filename), 'w') as txt_file:
                for shape in data['shapes']:
                    label = shape['label']
                    points = shape['points']

                    # Convert points to YOLO format
                    yolo_data = []
                    for point in points:
                        x = point[0] / img_width
                        y = point[1] / img_height
                        yolo_data.extend([x, y])

                    # Write the label and points to the txt file
                    txt_file.write(f"{label} {' '.join(map(str, yolo_data))}\n")

            print(f"Saved TXT file: {txt_filename} in {path}")
