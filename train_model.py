import os
import shutil
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
from ultralytics import YOLO

csv_file = '/home/hottiiiieeee/Desktop/object detction/OBJECT-DETECTION-main/labels_train.csv (1)/labels_train.csv (2)-(1)/labels_train.csv'
images_dir = '/home/hottiiiieeee/Desktop/object detction/OBJECT-DETECTION-main/images/train/images'
output_dir = '/home/hottiiiieeee/Desktop/object detction/OBJECT-DETECTION-main/OUTPUT'
epochs = 50
batch_size = 8
imgsz = 640

print("📦 Converting CSV to YOLO format...")

df = pd.read_csv(csv_file)

print("Columns in CSV file:", df.columns)

df['filename'] = df['frame'].apply(lambda x: f"{x}.jpg")

if 'class_id' not in df.columns:
    print("❌ CSV must contain a 'class_id' column.")
    exit()

class_names = df['class_id'].unique().tolist()
class_dict = {idx: name for idx, name in enumerate(class_names)}
df['class_id'] = df['class_id'].map(class_dict)

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
        print(f"⚠️ Image not found: {filename}")
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
print(f"YAML saved to: {yaml_path}")

print("🚀 Starting training...")
model = YOLO('yolov8n.pt')  # or yolov8s.pt
model.train(data=yaml_path, epochs=epochs, imgsz=imgsz, batch=batch_size)
print("✅ Training complete.")
