import cv2
from queue_analyzer import QueueAnalyzer

img = cv2.imread("test2.jpg")

analyzer = QueueAnalyzer(tile_size_m=0.3)

result = analyzer.analyze(img)

print("Людей:", result["people_count"])
print("Длина очереди (м):", result["queue_length_m"])

# рисуем
vis = img.copy()

vis = analyzer.draw(vis, result["people"])
vis = analyzer.draw_ruler(vis, result["people"], result["queue_length_m"])

vis = analyzer.resize_image(vis)

cv2.imshow("Result", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()