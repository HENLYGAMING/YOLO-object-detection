import cv2
import time
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.predict(source='https://www.youtube.com/watch?v=YfmnEiQdksE', stream=True)

for r in results:
  start_time = time.time()

  frame = r.plot()

  resized = cv2.resize(frame, (800, 640))

  # FPS counter
  end_time = time.time()
  epsilon = 1e-6
  current_fps = 1 / (end_time - start_time + epsilon)
  cv2.putText(resized, f"FPS: {int(current_fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

  cv2.imshow('YOLOv8 Detection', resized)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cv2.destroyAllWindows()
