import cv2 
import numpy as np
import os


camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    cv2.imshow("Imagem", frame)

    k = cv2.waitKey(60)
    if k == 27:
        break
