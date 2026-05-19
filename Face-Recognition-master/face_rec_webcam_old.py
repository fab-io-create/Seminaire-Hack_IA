from Yoda import FaceRecognition
import cv2
import numpy as np
import skimage.io as skio
import os
import os

def load_rgb(path):
    img = cv2.imread(os.path.join('./Personnes/', path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

video_capture = cv2.VideoCapture(0)
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
            # Transforme le nom de fichier en nom affiché : "XX_XX.jpg" → "XX XX"
            name = os.path.splitext(filename)[0]
            name = name.replace('_', ' ').replace('-', ' ')
            name = name.title()
            known_face_names.append(name)
        else:
            print(f"Aucun visage détecté dans {filename}, fichier ignoré.")
print(f"✅ {len(known_face_names)} personne(s) chargée(s) : {known_face_names}")

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = model.find_face_locations(rgb_small_frame)
        face_encodings = model.face_embeddings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = model.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                name = known_face_names[matches.index(True)]

            face_names.append(name)

    process_this_frame = not process_this_frame

    for d, name in zip(face_locations, face_names):
        top = d.top() * 4
        right = d.right() * 4
        bottom = d.bottom() * 4
        left = d.left() * 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()









