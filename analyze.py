from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

results = model("dataset/test/images")

for r in results:
    print("Image:", r.path)
    for box in r.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"  Class: {cls}, Confidence: {conf:.2f}, Box: {x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}")
