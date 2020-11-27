import cv2 as cv
import numpy as np
import face_recognition
import os
import glob

faces_encodings = []
faces_names = []
currentDir = os.getcwd()
path = os.path.join(currentDir, "rostos/")

lista = [f for f in glob.glob(path+"*.jpg")]
tamanhoLista = len(lista)
names = lista.copy()

for i in range(tamanhoLista):
    presets = face_recognition.load_image_file(lista[i])
    encoding = face_recognition.face_encodings(presets)[0]
    faces_encodings.append(encoding)
    names[i] = names[i].replace(currentDir, "")
    names[i] = names[i].replace(".jpg", "")
    names[i] = names[i].replace("rostos", "")
    faces_names.append(names[i])

face_locations = []
face_encodings = []
face_names = []

camera = cv.VideoCapture(0)
while True:

    _, frame = camera.read()
    smallFrame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgbSmallFrame = smallFrame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgbSmallFrame)
    face_encodings = face_recognition.face_encodings(
        rgbSmallFrame, face_locations)
    face_names = []
    for face in face_encodings:
        matches = face_recognition.compare_faces(faces_encodings, face)
        name = "Unknown"
        face_distances = face_recognition.face_distance(faces_encodings, face)
        bestMatch = np.argmin(face_distances)
        if matches[bestMatch]:
            name = faces_names[bestMatch]

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top = top * 4
        right = right * 4
        bottom = bottom * 4
        left = left * 4
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv.rectangle(frame, (left, bottom-35), (right, bottom), (0,0,255), cv.FILLED)
        cv.putText(frame, name, (left+6, bottom-6),
                    cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1)

    cv.imshow("Camera", frame)
    k = cv.waitKey(30)
    if k == 27:
        break

cv.destroyAllWindows()
camera.release()
