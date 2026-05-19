from Yoda import FaceRecognition
import cv2
import numpy as np
import os

def load_rgb(path):
    img = cv2.imread(os.path.join('./Personnes/', path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# --- Chargement des personnes ---
model = FaceRecognition()
known_face_encodings = []
known_face_names = []
PERSONS_DIR = './Personnes/'
valid_extensions = ('.jpg', '.jpeg', '.png')

for filename in sorted(os.listdir(PERSONS_DIR)):
    if filename.lower().endswith(valid_extensions):
        image = load_rgb(filename)
        encodings = model.face_embeddings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0].replace('_', ' ').replace('-', ' ').title()
            known_face_names.append(name)
        else:
            print(f"Aucun visage détecté dans {filename}, fichier ignoré.")

print(f"✅ {len(known_face_names)} personne(s) chargée(s) : {known_face_names}")

# --- Capture vidéo ---
CAM_W, CAM_H = 1280, 960
video_capture = cv2.VideoCapture(0, cv2.CAP_V4L2)
video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
video_capture.set(cv2.CAP_PROP_FPS, 30)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
actual_w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Résolution effective : {actual_w}×{actual_h}")

# Tout est calculé dynamiquement depuis la résolution réelle
DETECTION_SCALE = 0.25
SMALL_W = int(actual_w * DETECTION_SCALE)
SMALL_H = int(actual_h * DETECTION_SCALE)
BOX_SCALE = int(1 / DETECTION_SCALE)  # = 4 si DETECTION_SCALE=0.25, s'adapte automatiquement

font = cv2.FONT_HERSHEY_DUPLEX
face_locations = []
face_names = []
frame_count = 0
PROCESS_EVERY_N = 5

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_count += 1

    if frame_count % PROCESS_EVERY_N == 0:
        small = cv2.resize(frame, (SMALL_W, SMALL_H), interpolation=cv2.INTER_NEAREST)
        rgb_small = small[:, :, ::-1]

        new_locations = model.find_face_locations(rgb_small)

        if new_locations:
            encodings = model.face_embeddings(rgb_small, new_locations)
            new_names = []
            for enc in encodings:
                matches = model.compare_faces(known_face_encodings, enc)
                new_names.append(known_face_names[matches.index(True)] if True in matches else "Unknown")
            face_locations = new_locations
            face_names = new_names
        else:
            face_locations = []
            face_names = []

    for d, name in zip(face_locations, face_names):
        top = d.top() * BOX_SCALE
        right = d.right() * BOX_SCALE
        bottom = d.bottom() * BOX_SCALE
        left = d.left() * BOX_SCALE
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()