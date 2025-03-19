import cv2

# โหลด pre-trained Haar Cascade classifier สำหรับใบหน้า
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# เปิดกล้อง
cap = cv2.VideoCapture(0)

while True:
    # อ่านภาพจากกล้อง
    ret, frame = cap.read()

    # แปลงภาพเป็นขาวดำ เพื่อเพิ่มความเร็วในการตรวจจับ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้าในภาพ
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # วาดสี่เหลี่ยมรอบ ๆ ใบหน้า
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # แสดงผลลัพธ์
    cv2.imshow('Face Detection', frame)

    # หากกด 'q' จะหยุดการทำงาน
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ปิดกล้องและหน้าต่าง
cap.release()
cv2.destroyAllWindows()
