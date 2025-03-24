import cv2
import torch
from ultralytics import YOLO

# ตรวจสอบว่าใช้ GPU ได้หรือไม่
device = "cuda" if torch.cuda.is_available() else "cpu"

# โหลดโมเดล YOLOv8 (เฉพาะการตรวจจับวัตถุ)
model = YOLO("yolov8n.pt").to(device)  # ใช้เวอร์ชัน 'n' (Nano) เพื่อลดโหลดการประมวลผล

# ตั้งค่าการประมวลผล
CONFIDENCE_THRESHOLD = 0.4  # ค่าความมั่นใจขั้นต่ำสำหรับการตรวจจับ
IMG_SIZE = 1280  # ลดขนาดภาพให้เล็กลงเพื่อเพิ่มความเร็ว
FRAME_SKIP = 1  # ข้ามเฟรมเพื่อลดโหลด CPU

# เปิดกล้องหรือวิดีโอ
cap = cv2.VideoCapture("./1.mp4")  # ใส่พาธวิดีโอถ้าต้องการใช้ไฟล์แทนกล้อง

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue  # ข้ามเฟรมตามค่า FRAME_SKIP

    # Resize ภาพเพื่อเพิ่มความเร็ว
    frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # รันการตรวจจับ
    results = model(frame_resized, conf=CONFIDENCE_THRESHOLD, classes=[0])  # classes=[0] หมายถึงตรวจจับเฉพาะคน

    # วาดกรอบรอบตัวบุคคล
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # พิกัดกรอบ
            conf = box.conf[0].item()  # ค่าความมั่นใจ
            label = f"Person {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # แสดงผล
    cv2.imshow("YOLOv8 Person Detection", frame)

    # กด 'q' เพื่อออก
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
