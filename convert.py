from ultralytics import YOLO
import cv2

# โหลดโมเดล YOLOv8
model = YOLO("yolov8n.pt")  # สามารถเปลี่ยนชื่อโมเดลเป็น yolov8s.pt, yolov8m.pt, หรือ yolov8l.pt ได้

cap = cv2.VideoCapture(0)  # ใช้กล้องเว็บแคม

line_x = 600
people_in = 0
people_out = 0
trackers = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    height, width, _ = frame.shape
    
    # ทำการตรวจจับวัตถุ
    results = model(frame)  # ทำการตรวจจับ

    # ดึงผลการตรวจจับที่ได้
    boxes = []
    confidences = []
    class_ids = []
    
    # ใช้ผลลัพธ์จาก results[0].boxes
    for result in results[0].boxes:
        x, y, w, h = result.xywh[0].tolist()  # แยกค่า x, y, width, height
        conf = result.conf[0].item()  # ดึง confidence
        cls = int(result.cls[0].item())  # ดึง class id และแปลงเป็น int
        
        if conf > 0.5 and cls == 0:  # ตรวจจับเฉพาะ "person" (class ID = 0)
            x1 = int(x - w / 2)  # คำนวณพิกัดมุมบนซ้าย
            y1 = int(y - h / 2)
            x2 = int(x + w / 2)  # คำนวณพิกัดมุมล่างขวา
            y2 = int(y + h / 2)
            boxes.append([x1, y1, x2, y2])  # บันทึกพิกัดกล่อง
            confidences.append(conf)
            class_ids.append(cls)

    # ใช้ Non-Maximum Suppression (NMS) เพื่อลดการตรวจจับที่ซ้ำ
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    detected_people = []
    
    if len(indexes) > 0:
        for i in indexes.flatten():
            x1, y1, x2, y2 = boxes[i]
            center_x = (x1 + x2) // 2
            detected_people.append((x1, y1, x2, y2, center_x))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    new_trackers = []
    for x1, y1, x2, y2, center_x in detected_people:
        new_trackers.append((x1, y1, x2, y2, center_x))
    
    for (old_x1, old_y1, old_x2, old_y2, old_center_x) in trackers:
        closest = min(new_trackers, key=lambda p: abs(p[4] - old_center_x), default=None)
        if closest:
            _, _, _, _, new_center_x = closest
            if old_center_x < line_x and new_center_x >= line_x:
                people_in += 1
            elif old_center_x > line_x and new_center_x <= line_x:
                people_out += 1
            new_trackers.remove(closest)
    
    trackers = new_trackers
    
    # วาดเส้นบนแกน X
    cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 2)
    cv2.putText(frame, f"In: {people_in} Out: {people_out}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
