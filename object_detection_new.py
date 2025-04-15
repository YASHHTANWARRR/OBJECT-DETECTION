#libraries 
import numpy as np 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt 
import sklearn
import PIL
import os 
import pathlib


from sklearn.utils import shuffle
from matplotlib.patches import Rectangle
import warnings
from ultralytics import YOLO
from PIL import Image
from IPython.display import display

warnings.simplefilter('ignore')

#dataset ready
df = pd.read_csv('/home/hottiiiieeee/Desktop/object detction/OBJECT-DETECTION-main/labels_train.csv (1)')
df = shuffle(df)
df.head 

classes = df.classes_id_unique()
print (classes)

labels = { 1 : 'car',2 : 'truck' , 3 : 'person' , 4 : 'bicycle' , 5 : 'traffic light'}

#labels and boxes the detection 

boxes = {}
labels = {}

base_path =' /home/hottiiiieeee/Desktop/object detction/OBJECT-DETECTION-main/labels_train.csv (1)'

for classes_id in classes:
    first_row = df[df['classes_id'] == classes_id].iloc[0]

    images[classes_id] = cv2.imread[base_path +first_row['frame']]
    boxes[classes_id] = [first_row['xmin'],first_row['xmax'],first_row['ymin'],first_row['ymax']]

for i in classes:
    xmin,xmax,ymin,ymax = boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]
    plt.figure(figsize=(8,10))
    plt.label('label' + labels[i])
    plt.imshow(images[i])
    plt.gca().add_patch(plt.Rectangle((xmin,ymin)k,xmax-xmin,ymax-ymin,color='yellow',fill=False,linewwidth=2))

    plt.show()

#model
model = YOLO('yolov8m.pt')
result  = results[0]
box = result.boxes[0]

for result in results :
    boxes = result.boxes
    masks = result.masks
    probs = result.prob
    
cords = box.xyxy[0].tolist()
class_id = box.cls[0].item()
conf = box.conf[0].item()
print("Object type:", class_id)
print("Coordinates:", cords)
print("Probability:", conf)

for box in result.boxes:
    class_id = result.names[box.cls[0].item()]
    cords = box.xyxy[0].tolist()
    cords = [round(x) for x in cords]
    conf = round(box.conf[0].item(), 2)
    print("Object type:", class_id)
    print("Coordinates:", cords)
    print("Probability:", conf)
    print("---")

results1 = model.predict(source="/kaggle/input/self-driving-cars/images/1478020211690815798.jpg",
              save=True, conf=0.2,iou=0.5)

Results = results1[0]

# Plotting results
plot = results1[0].plot()
plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
display(Image.fromarray(plot))

def calculate_iou(box1, box2):
   
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    
   
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

     box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_p - x1_p) * (y2_p - y1_p)
    
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou

image_name = "1478020211690815798.jpg"
ground_truth = df[df['frame'] == image_name].reset_index(drop=True)


gt_boxes = ground_truth[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
gt_labels = ground_truth['class_id'].map(labels).tolist() 


pred_boxes = [box.xyxy[0].tolist() for box in results1[0].boxes]
pred_labels = [results1[0].names[box.cls[0].item()] for box in results1[0].boxes]
pred_confs = [box.conf[0].item() for box in results1[0].boxes]


iou_threshold = 0.5  
tp, fp, fn = 0, 0, len(gt_boxes)

matched_gt = set() 


for pred_box, pred_label, pred_conf in zip(pred_boxes, pred_labels, pred_confs):
    best_iou = 0
    best_gt_idx = -1
    
   
    for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
        if gt_idx in matched_gt:
            continue
        if pred_label != gt_label:
            continue
        iou = calculate_iou(pred_box, gt_box)
        if iou > best_iou:
            best_iou = iou
            best_gt_idx = gt_idx
    
    
    if best_iou >= iou_threshold:
        tp += 1
        fn -= 1  
        matched_gt.add(best_gt_idx)

    else:
        fp += 1


precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0


print(f"\nPerformance Metrics for {image_name}:")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


map_score = precision * recall 
print(f"mAP@0.5 (approximated): {map_score:.2f}")
