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











                                    






