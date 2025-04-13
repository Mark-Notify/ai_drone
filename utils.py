
# ======================== utils.py ========================
import face_recognition
import cv2
import numpy as np
import os

def load_known_faces(folder_path):
    known_faces = []
    known_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = face_recognition.load_image_file(os.path.join(folder_path, filename))
            encodings = face_recognition.face_encodings(image)
            if encodings:
                known_faces.append(encodings[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_faces, known_names

def recognize_faces_in_frame(face_encoding, known_faces, known_names):
    matches = face_recognition.compare_faces(known_faces, face_encoding)
    face_distances = face_recognition.face_distance(known_faces, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        return known_names[best_match_index]
    return "Unknown"

def draw_name(frame, name, location):
    top, right, bottom, left = location
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

def move_drone_towards(name):
    print(f"กำลังบินเข้าหา {name}...")
