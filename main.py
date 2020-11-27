import cv2 as cv 
import numpy as np
import os

camera = cv.VideoCapture(0)

while True:
    _, frame = camera.read()
    framehsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    menorverde = np.array([57,147,129])
    maiorverde = np.array([255,255,255])

    maskverde = cv.inRange(framehsv,menorverde,maiorverde)
    resultadoverde = cv.bitwise_and(frame,frame,mask=maskverde)
    framecinzaverde = cv.cvtColor(resultadoverde, cv.COLOR_BGR2GRAY)
    _, threshverde = cv.threshold(framecinzaverde,3,255,cv.THRESH_BINARY)
    contornosverde, _ = cv.findContours(threshverde, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for contorno in contornosverde:
        (x,y,w,h) = cv.boundingRect(contorno)


    cv.imshow("Imagem", frame)
    cv.imshow("Imagemhsv",framehsv)
    k = cv.waitKey(60)
    if k == 27:
        break

cv.destroyAllWindows()
camera.release
