import cv2
import numpy as np
from roboflow import Roboflow
import supervision as sv
import time
import os


api_key = os.getenv("ROBOFLOW_API_KEY")
rf = Roboflow(api_key=api_key)

model = rf.workspace("asldetect").project("american-sign-language-alphabet-miocm-lxha0").version(6).model

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Failed to open camera.")
    exit()

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

print("Starting ASL detection... Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Inference
    start = time.time()
    try:
        result = model.predict(frame, confidence=40, overlap=30).json()
    except Exception as e:
        print(f"Inference error: {e}")
        continue
    end = time.time()
    print(f"Inference time: {end - start:.2f}s")

    predictions = result.get("predictions", [])
    boxes = []
    confidences = []
    class_ids = []
    class_names = []

    for pred in predictions:
        x1 = pred["x"] - pred["width"] / 2
        y1 = pred["y"] - pred["height"] / 2
        x2 = pred["x"] + pred["width"] / 2
        y2 = pred["y"] + pred["height"] / 2
        boxes.append([x1, y1, x2, y2])
        confidences.append(pred["confidence"])
        class_ids.append(pred.get("class_id", 0))
        class_names.append(pred["class"])

    if boxes:
        detections = sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(confidences),
            class_id=np.array(class_ids),
            data={"class_name": class_names}
        )

        labels = [f"{class_names[i]} {confidences[i]:.2f}" for i in range(len(class_names))]
        frame = box_annotator.annotate(scene=frame, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    cv2.imshow("ASL Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
