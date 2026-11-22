from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # "n" = nano, the smallest and fastest size
results = model("https://ultralytics.com/images/bus.jpg")

result = results[0]
for box in result.boxes:
    class_name = result.names[int(box.cls[0])]
    confidence = float(box.conf[0])
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    print(f"{class_name:10s} conf={confidence:.2f} box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")

result.save(filename="bus_annotated.jpg")