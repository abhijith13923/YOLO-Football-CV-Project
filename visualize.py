from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train2/weights/best.pt")
class_names = model.names

results = model("dataset/test/images")

for r in results:
    img = cv2.imread(r.path)

    for box in r.boxes:
        cls = int(box.cls)
        conf = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        label = f"{class_names[cls]} {conf:.2f}"

        # draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # draw label
        cv2.putText(
            img,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    cv2.imshow("YOLO Predictions", img)
    cv2.waitKey(0)   # press any key for next image

cv2.destroyAllWindows()
