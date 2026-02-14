from ultralytics import YOLO

mode = YOLO('yolo11n.pt')

mode.train(
    data = 'dataset/data.yaml',
    epochs = 15,
    imgsz = 640,
    batch = 16,
    device = 'cpu'
)