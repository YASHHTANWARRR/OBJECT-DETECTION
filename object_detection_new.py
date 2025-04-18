import os
import shutil
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
import time
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# ==== CONFIG ====
csv_file = '/path/to/annotations.csv'
images_dir = '/path/to/images'
output_dir = '/path/to/yolo_dataset'
epochs = 50
batch_size = 8
imgsz = 640
# ================

# ==== STEP 1: Convert CSV to YOLO format ====
print("📦 Converting CSV to YOLO format...")

df = pd.read_csv(csv_file)
class_names = df['class_name'].unique().tolist()
class_dict = {name: idx for idx, name in enumerate(class_names)}
df['class_id'] = df['class_name'].map(class_dict)

train_imgs, val_imgs = train_test_split(df['filename'].unique(), test_size=0.2, random_state=42)
df['split'] = df['filename'].apply(lambda x: 'train' if x in train_imgs else 'val')

for split in ['train', 'val']:
    os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)

for filename in tqdm(df['filename'].unique(), desc="Processing images"):
    sub_df = df[df['filename'] == filename]
    split = sub_df['split'].values[0]
    image_path = os.path.join(images_dir, filename)

    try:
        with Image.open(image_path) as img:
            img_w, img_h = img.size
    except FileNotFoundError:
        print(f"Image not found: {filename}")
        continue

    label_lines = []
    for _, row in sub_df.iterrows():
        class_id = row['class_id']
        xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
        x_center = (xmin + xmax) / 2 / img_w
        y_center = (ymin + ymax) / 2 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    label_path = os.path.join(output_dir, 'labels', split, os.path.splitext(filename)[0] + '.txt')
    with open(label_path, 'w') as f:
        f.write('\n'.join(label_lines))

    shutil.copy(image_path, os.path.join(output_dir, 'images', split, filename))

yaml_dict = {
    'train': os.path.abspath(os.path.join(output_dir, 'images', 'train')),
    'val': os.path.abspath(os.path.join(output_dir, 'images', 'val')),
    'nc': len(class_names),
    'names': class_names
}

yaml_path = os.path.join(output_dir, 'data.yaml')
with open(yaml_path, 'w') as f:
    yaml.dump(yaml_dict, f)

print("✅ CSV converted. YOLO dataset ready.")
print("Class mapping:", class_dict)
print(f"YAML: {yaml_path}")

# ==== STEP 2: Train the model ====
print("🚀 Starting training...")
model = YOLO('yolov8n.pt')  # Use yolov8s.pt or yolov8m.pt if needed
model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, batch=batch_size)
print("✅ Training done.")

# ==== STEP 3: Real-time detection from PiCamera ====
best_model_path = 'runs/detect/train/weights/best.pt'
print(f"🎯 Loading trained model: {best_model_path}")
trained_model = YOLO(best_model_path)

print("🎥 Starting PiCamera detection... Press 'q' to quit.")
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)

try:
    while True:
        start_time = time.time()
        frame = picam2.capture_array()
        results = trained_model.predict(source=frame, imgsz=imgsz, conf=0.25)
        im_result = results[0].plot()
        cv2.imshow("YOLOv8 Detection", im_result)
        print(f"FPS: {1.0 / (time.time() - start_time):.2f}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("🛑 Detection stopped.")
