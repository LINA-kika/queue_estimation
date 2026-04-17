import cv2
import numpy as np
from ultralytics import YOLO


class QueueAnalyzer:
    def __init__(self, model_path="yolov8n.pt", tile_size_m=0.6):
        self.model = YOLO(model_path)
        self.tile_size_m = tile_size_m  # размер плитки (метры)

    def detect_people(self, image):
        results = self.model(image)[0]

        people = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                people.append({
                    "box": (x1, y1, x2, y2),
                    "center": (cx, cy)
                })

        return people

    def sort_queue(self, people):
        """
        Сортируем людей по оси X (очередь слева направо)
        """
        return sorted(people, key=lambda p: p["center"][0])

    def estimate_pixel_to_meter(self, image):
        """
        Оцениваем масштаб через плитку.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                                minLineLength=50, maxLineGap=10)

        horizontal_lines = []

        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                if abs(y1 - y2) < 10:  # горизонтальные
                    length = abs(x2 - x1)
                    horizontal_lines.append(length)

        if len(horizontal_lines) == 0:
            return 100  # fallback

        avg_tile_px = np.median(horizontal_lines)

        px_to_meter = self.tile_size_m / avg_tile_px
        return px_to_meter

    def estimate_queue_length(self, people, px_to_meter):
        if len(people) < 2:
            return 0

        total_px = 0

        for i in range(len(people) - 1):
            x1, _ = people[i]["center"]
            x2, _ = people[i + 1]["center"]

            dist = abs(x2 - x1)
            total_px += dist

        length_m = total_px * px_to_meter
        return round(length_m, 2)

    def draw(self, image, people):
        for p in people:
            x1, y1, x2, y2 = p["box"]
            cx, cy = p["center"]

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(image, (cx, cy), 5, (0, 0, 255), -1)

        return image
    
    def resize_image(self, img, max_width=800):
        h, w = img.shape[:2]
        
        if w > max_width:
            scale = max_width / w
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
        
        return img
    
    
    def draw_ruler(self, image, people, queue_length_m):
        if len(people) < 2:
            return image

        # первая и последняя точки
        x1, y1 = people[0]["center"]
        x2, y2 = people[-1]["center"]

        # линия 
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # текст
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        text = f"{queue_length_m} m"
        cv2.putText(
            image,
            text,
            (mid_x - 50, mid_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

        return image

    def analyze(self, image):
        people = self.detect_people(image)
        people = self.sort_queue(people)

        px_to_meter = self.estimate_pixel_to_meter(image)
        length = self.estimate_queue_length(people, px_to_meter)

        return {
            "people_count": len(people),
            "queue_length_m": length,
            "people": people
        }