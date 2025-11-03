import os
import torch
import torchvision.transforms as T
import torchvision.ops as ops
from PIL import Image, ImageDraw
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import numpy as np

import pathlib

def load_and_shuffle_labels(csv_path):
    df = pd.read_csv(csv_path)
    df = shuffle(df, random_state=42)
    return df

def visualize_first_instances(df, images_dir, class_map):
    classes = df['class_id'].unique()
    for class_id in classes:
        row = df[df['class_id'] == class_id].iloc[0]
        img_path = os.path.join(images_dir, f"{row['frame']}.jpg")
        if not os.path.isfile(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue
        img = Image.open(img_path).convert("RGB")
        xmin, xmax, ymin, ymax = row[['xmin', 'xmax', 'ymin', 'ymax']]
        draw = ImageDraw.Draw(img)
        draw.rectangle([xmin, ymin, xmax, ymax], outline='yellow', width=3)
        plt.figure(figsize=(8, 6))
        plt.title(f"Label: {class_map.get(class_id, class_id)}")
        plt.imshow(img)
        plt.axis('off')
        plt.show()

def load_yolov8_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model = checkpoint['model'].float()
    model.eval()
    return model

def preprocess_image(image_path, img_size=160):
    img = Image.open(image_path).convert('RGB')
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_t = transform(img).unsqueeze(0)
    return img_t, img

def non_max_suppression(prediction, conf_thresh=0.25, iou_thresh=0.5):
    boxes = prediction[:, :4]
    scores = prediction[:, 4]
    keep = ops.nms(boxes, scores, iou_thresh)
    return prediction[keep]

# def predict_with_pytorch(model, image_tensor, conf_thresh=0.25, iou_thresh=0.5):
#     with torch.no_grad():
#         preds = model(image_tensor)
#     pred = preds[0]
#     xy = pred[:, :4]
#     x1 = xy[:, 0] - xy[:, 2] / 2
#     y1 = xy[:, 1] - xy[:, 3] / 2
#     x2 = xy[:, 0] + xy[:, 2] / 2
#     y2 = xy[:, 1] + xy[:, 3] / 2
#     boxes = torch.stack([x1, y1, x2, y2], dim=1)
#     conf = pred[:, 4]
#     class_probs = pred[:, 5:]
#     class_conf, class_pred = torch.max(class_probs, dim=1)
#     scores = conf * class_conf
#     keep = scores > conf_thresh
#     filtered_boxes = boxes[keep]
#     filtered_scores = scores[keep]
#     filtered_classes = class_pred[keep].float()
#     detections = torch.cat([filtered_boxes, filtered_scores.unsqueeze(1), filtered_classes.unsqueeze(1)], dim=1)
#     detections = non_max_suppression(detections, conf_thresh, iou_thresh)
#     return detections

def predict_with_pytorch(model, image_tensor, conf_thresh=0.25, iou_thresh=0.5):
    with torch.no_grad():
        preds = model(image_tensor)
    pred = preds[0]  # Predictions for batch item 0
    # pred shape is [num_detections, 85]

    xy = pred[:, :4]
    x1 = xy[:, 0] - xy[:, 2] / 2
    y1 = xy[:, 1] - xy[:, 3] / 2
    x2 = xy[:, 0] + xy[:, 2] / 2
    y2 = xy[:, 1] + xy[:, 3] / 2
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    conf = pred[:, 4]
    class_probs = pred[:, 5:]
    class_conf, class_pred = torch.max(class_probs, dim=1)
    scores = conf * class_conf

    keep = scores > conf_thresh
    boxes = boxes[keep]
    scores = scores[keep]
    classes = class_pred[keep]

    detections = torch.cat([boxes, scores.unsqueeze(1), classes.unsqueeze(1).float()], dim=1)
    detections = non_max_suppression(detections, conf_thresh, iou_thresh)
    return detections


def draw_detections(image, detections, class_map):
    draw = ImageDraw.Draw(image)
    for x1, y1, x2, y2, conf, cls in detections:
        cls = int(cls.item())
        conf = conf.item()
        draw.rectangle([x1, y1, x2, y2], outline='yellow', width=3)
        draw.text((x1, y1), f"{class_map.get(cls, cls)} {conf:.2f}", fill='yellow')

if __name__ == '__main__':
    csv_file = '/Users/birba/OneDrive/Documents/OBJECT-DETECTION/labels_train.csv'
    images_dir = '/Users/birba/OneDrive/Documents/OBJECT-DETECTION/images'
    sample_image = os.path.join(images_dir, 'test.jpg')
    model_path = 'yolov8m.pt'
    class_map = {1: 'car', 2: 'truck', 3: 'person', 4: 'bicycle', 5: 'traffic light'}
    df = load_and_shuffle_labels(csv_file)
    print("Classes in dataset:", df['class_id'].unique())
    visualize_first_instances(df, images_dir, class_map)
    model = load_yolov8_model(model_path)
    img_tensor, pil_img = preprocess_image(sample_image, img_size=160)
    detections = predict_with_pytorch(model, img_tensor)
    draw_detections(pil_img, detections, class_map)
    pil_img.show()
    
img_path = pathlib.Path(images_dir) / f"{row['frame']}.jpg"
img_path = str(img_path)
