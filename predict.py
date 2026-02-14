from ultralytics import YOLO

model = YOLO("runs/detect/train2/weights/best.pt")

model.predict(
    source="dataset/test/images",
    save=True,
    conf=0.4
)
