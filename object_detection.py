import numpy as np 
import pandas as pd 
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#importing data from kaggle
import kagglehub

# Download latest version
path = kagglehub.dataset_download("sshikamaru/car-object-detection")
print (os.path.join((C:\Users\birba\OneDrive\Desktop\autonomous vehicle\data))) 
model = yolo('yolov8u.pt')
data_imagepath="C:\Users\birba\OneDrive\Desktop\autonomous vehicle\data\testing_images"
image = cv2.imread(data_imagepath)
results = model (image)

for result in results :
    for box in result.boxes:
        x1,y1,x2,y2 = map(int,box.xyxy[0]) #(x1,y1 is the top left corner),(x2,y2 is the bottom right corner)
        label = f"{model.names[int(box.cls[0])]}:{box.conf[0]:.2f}"
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),2)#drawiing a box around the object
        cv2.putText(image,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX),0.5,(0,255,0),2)#placing a green label on the box 

#saving the final image to disk
output_path = "output.jpg"
cv2.inwrite(output_path,image)

output_image = cv2.imread(output_path)
cv2.imshow("YOLOv8 Detection",output_image)
cv2.wwaitkey(0)
cv2.destroyAllwindows

img = mpimg.imread('C:\Users\birba\OneDrive\Desktop\autonomous vehicle\data\testing_images')
plt.imshow(img)
plt.axis('off')
plt.show()

#applying NON - NMS approach
from ultralytics import YOLO
model = ('yolov8.pt')

dataset_imagepath1 = "C:\Users\birba\OneDrive\Desktop\autonomous vehicle\data\testing_images"

image1= cv2.imread(dataset_imagepath1)

result_no_nms = model(image1,nms=False)
result_nms = model(image1,nms=True)

image_no_nms=image.copy()

for result in result_no_nms:
    for box in result.boxes:
        x1,y1,x2,y2 = map (int,box.xyxy[0])
        label = f"{model.names(box.cls[0])}:{box.config[0]:.2f}"
        cv2.rectangle(image_no_nms,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(image_no_nms,label1,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2) 

image_with_nms=image.copy()

for result in result_no_nms:
    for box in result.boxes:
        x1,y1,x2,y2 = map (int,box.xyxy[0])
        label = f"{model.names(box.cls[0])}:{box.config[0]:.2f}"
        cv2.rectangle(image_with_nms,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(image_with_nms,label1,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2) 

plt.figur(figaize=(10,5))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image_no_nms,cv2.COLOR_BGR2RGB))
plt.title('Without NMS')
plt.axis('off')

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image_with_nms,cv2.BGR2RGB))
plt.title('With NMS')
plt.axis('off')
plt.show()

def get_iou(a,b):#intersection over union,important for nms
    ax1,ay1,ax2,ay2=a
    bx1,by1,bx2,by2=b
    left = (ax1,bx1)
    right= (ax2,bx2)
    top =(ay1,by1)
    bottom = (ay2,by2)
    if (left>right) and (top > bottom):
        return "no overlap"
    a_area = abs(ax2-ax1)*abs(bx2-bx1)
    b_area = abs(ay2-ay1)*abs(by2-by1)
    intersection = abs(left-right)*abs(top-bottom)
    union_area = a_area + b_area - intersection
    final_iou = intersection / union_area
    return final_iou

#removing overlapping bxes and keeping the originals
def nms(a,threshold = 0.5):
    
