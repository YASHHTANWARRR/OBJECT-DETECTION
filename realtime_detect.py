import cv2
import time
from ultralytics import YOLO
from picamera2 import Picamera2

model_path = '/home/hottiiiieeee/Desktop/object detection/OBJECT-DETECTION-main/object_detection_new.py'
conf_threshold = 0.25
iou_threshold = 0.5
imgsz = 320

def setup_camera(resolution=(640, 480)):
    cam = Picamera2()
    config = cam.create_preview_configuration(main={"size": resolution})
    cam.configure(config)
    cam.start()
    time.sleep(2)
    return cam

def real_time_detection(model_path, conf=0.25, iou=0.5, imgsz=320):
    model = YOLO(model_path,task='detect')
    cam = setup_camera()
    try:
        while True:
            frame = cam.capture_array()
            results = model.predict(source=frame, imgsz=imgsz, conf=conf, iou=iou, device='cpu', verbose=False)
            annotated = results[0].plot()
            cv2.imshow("YOLOv8 Real-Time Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cam.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    real_time_detection(model_path, conf_threshold, iou_threshold, imgsz)

