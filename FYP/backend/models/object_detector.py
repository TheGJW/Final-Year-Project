from ultralytics import YOLO

class ObjectDetector:
    # initializing the yolo model
    def __init__(self):
        self.model = YOLO("yolov8n.pt")
    # function to perform object detection on image
    def detect(self, image_path, conf_threshold=0.1):

        # running inference on the input image
        results = self.model(image_path)
        # list to store the detected bounding boxes
        boxes = []

        for r in results:
            for box in r.boxes:
                # extracting the confidence score of detection
                confidence = float(box.conf[0])
                # skip detections below confidence threshold
                if confidence < conf_threshold:
                    continue
                # extracting the bounding boxes coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                boxes.append({
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2,
                    "confidence": confidence,
                    "class_id": int(box.cls[0]),
                    "class_name": self.model.names[int(box.cls[0])]
                })

        return boxes