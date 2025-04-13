import os
import cv2
import json
import numpy as np
import face_recognition
from ultralytics import YOLO
import airsim
import uuid

# พาธฐานข้อมูล
BASE_PATH = "faces_db"
DB_FILE = "faces.json"

# โหลดฐานข้อมูลใบหน้า
if os.path.exists(DB_FILE):
    with open(DB_FILE, "r") as f:
        known_faces = json.load(f)
else:
    known_faces = {}

# โหลดใบหน้า (encoding)
face_encodings = []
face_names = []

for name, paths in known_faces.items():
    for path in paths:
        img = face_recognition.load_image_file(path)
        enc = face_recognition.face_encodings(img)
        if enc:
            face_encodings.append(enc[0])
            face_names.append(name)

# YOLOv8
model = YOLO("yolov8n.pt")

# AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()

# สำหรับการคลิกเลือกคน
selected_box = None
bounding_boxes = []
face_labels = []

def move_to_target(x, y, w, h, img_shape):
    img_center_x = img_shape[1] // 2
    target_x = x + w // 2

    if target_x < img_center_x - 50:
        client.moveByVelocityAsync(-0.5, 0, 0, 1).join()
    elif target_x > img_center_x + 50:
        client.moveByVelocityAsync(0.5, 0, 0, 1).join()
    else:
        client.moveByVelocityAsync(0, 0.5, 0, 1).join()

def mouse_callback(event, x, y, flags, param):
    global selected_box
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, (x1, y1, x2, y2) in enumerate(bounding_boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_box = (x1, y1, x2, y2)
                print(f"เลือก: {face_labels[i]}")
                break

cv2.namedWindow("Drone View")
cv2.setMouseCallback("Drone View", mouse_callback)

try:
    while True:
        raw = client.simGetImage("0", airsim.ImageType.Scene)
        if raw is None:
            continue

        img1d = np.frombuffer(bytearray(raw), dtype=np.uint8)
        frame = cv2.imdecode(img1d, cv2.IMREAD_COLOR)

        # YOLO ตรวจจับ
        results = model(frame)[0]
        bounding_boxes.clear()
        face_labels.clear()

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bounding_boxes.append((x1, y1, x2, y2))

                # ตัดภาพใบหน้า
                face_img = frame[y1:y2, x1:x2]
                rgb_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                encs = face_recognition.face_encodings(rgb_face)

                if encs:
                    matches = face_recognition.compare_faces(face_encodings, encs[0])
                    name = "Unknown"

                    if True in matches:
                        name = face_names[matches.index(True)]
                    else:
                        cv2.imshow("Unknown Face", face_img)
                        cv2.waitKey(1)
                        name = input("ไม่รู้จัก ใส่ชื่อ: ").strip()

                        folder = os.path.join(BASE_PATH, name)
                        os.makedirs(folder, exist_ok=True)
                        filename = f"{uuid.uuid4().hex[:8]}.jpg"
                        filepath = os.path.join(folder, filename)
                        cv2.imwrite(filepath, face_img)

                        known_faces.setdefault(name, []).append(filepath)
                        face_encodings.append(encs[0])
                        face_names.append(name)

                        with open(DB_FILE, "w") as f:
                            json.dump(known_faces, f, indent=2)

                    face_labels.append(name)
                    cv2.putText(frame, name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    face_labels.append("Unknown")

                color = (0, 255, 0) if (x1, y1, x2, y2) != selected_box else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # บินเข้าใกล้เป้า
        if selected_box:
            x1, y1, x2, y2 = selected_box
            move_to_target(x1, y1, x2 - x1, y2 - y1, frame.shape)

        cv2.imshow("Drone View", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    pass

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
cv2.destroyAllWindows()
