import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from ultralytics import YOLO
from PIL import Image

csv_file = '/home/hottiiiieeee/Desktop/archive (2)/labels_train.csv'
images_dir = '/home/hottiiiieeee/Desktop/archive (2)/images'
sample_image = os.path.join(images_dir, '1478019952686311006.jpg')
best_model = 'yolov8m.pt'

def load_and_shuffle_labels(csv_path):
    df = pd.read_csv(csv_path)
    df = shuffle(df, random_state=42)
    return df

def visualize_first_instances(df, images_dir, class_map, max_per_class=1):
    classes = df['class_id'].unique()
    for class_id in classes:
        row = df[df['class_id'] == class_id].iloc[0]
        img_path = os.path.join(images_dir, f"{row['frame']}.jpg")
        if not os.path.isfile(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        xmin, xmax, ymin, ymax = row[['xmin', 'xmax', 'ymin', 'ymax']]
        plt.figure(figsize=(8, 6))
        plt.title(f"Label: {class_map.get(class_id, class_id)}")
        plt.imshow(img)
        plt.gca().add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                          edgecolor='yellow', facecolor='none', linewidth=2)
        )
        plt.axis('off')
        plt.show()

def predict_and_display(model_path, image_path, imgsz=320, conf=0.25, iou=0.5):
    model = YOLO(model_path)
    results = model.predict(source=image_path, imgsz=imgsz, conf=conf, iou=iou)
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            coord = [round(v) for v in box.xyxy[0].tolist()]
            confidence = round(box.conf[0].item(), 2)
            name = result.names.get(cls_id, cls_id)
            print(f"Object: {name}, Coordinates: {coord}, Confidence: {confidence}")
    annotated = results[0].plot()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    Image.fromarray(annotated).show()

if __name__ == '__main__':
    class_map = {1: 'car', 2: 'truck', 3: 'person', 4: 'bicycle', 5: 'traffic light'}
    df = load_and_shuffle_labels(csv_file)
    print("Classes in dataset:", df['class_id'].unique())
    visualize_first_instances(df, images_dir, class_map)
    print(f"\nRunning detection on: {sample_image}")
    predict_and_display(best_model, sample_image)
