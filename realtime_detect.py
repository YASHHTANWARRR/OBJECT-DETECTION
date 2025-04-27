import time
import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

model_path = '/home/hottiiiieeee/Desktop/object detction/OBJECT-DETECTION-main/OUTPUT/runs/detect/train/weights/best.pt'
imgsz = 640

# === STEP 1: Load trained model ===
print(f"🎯 Loading trained model from: {model_path}")
model = YOLO(model_path)

print("🎥 Starting PiCamera detection... Press 'q' to quit.")
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)

try:
    while True:
        start_time = time.time()
        frame = picam2.capture_array()
        results = model.predict(source=frame, imgsz=imgsz, conf=0.25)
        im_result = results[0].plot()
        cv2.imshow("YOLOv8 Detection", im_result)
        print(f"FPS: {1.0 / (time.time() - start_time):.2f}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("🛑 Detection stopped.")
