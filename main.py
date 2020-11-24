import cv2 
import numpy as np
import os


camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    framehsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    menorverde = np.array[57,147,129]
    maiorverde = np.array[255,255,255]

    maskverde = cv2.inRange(framehsv,menorverde,maiorverde)
    resultadoverde = cv2.bitwise_and(frame,frame,mask=maskverde)
    framecinzaverde = cv2.cvtColor(resultadoverde, cv.COLOR_BGR2GRAY)
    _, threshverde = cv2.threshold(framecinzaverde,3,255,cv.THRESH_BINARY)
    contornosverde, _ = cv2.findContours(threshverde, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    afericao = cv2.rectangle(contornosverde)


    cv2.imshow("Imagem", frame)
    cv2.imshow("Imagemhsv",framehsv)
    k = cv2.waitKey(60)
    if k == 27:
        break

cv2.destroyAllWindows()
camera.release
