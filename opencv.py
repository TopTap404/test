import cv2

cap = cv2.VideoCapture('./1.mp4')

# โหลด Haarcascade สำหรับตรวจจับคน
human_cascade = cv2.CascadeClassifier('./haarcascade_upperbody.xml')

# ตรวจสอบว่าโหลดไฟล์สำเร็จหรือไม่
if human_cascade.empty():
    print("Error: Haarcascade file not loaded properly. Check the XML file path!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    # แปลงเป็นขาวดำเพื่อลดโหลดการประมวลผล
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับคนในภาพ
    humans = human_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 60))

    # วาดกรอบรอบตัวบุคคล
    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Haarcascade People Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()