import cv2 as cv 
import numpy as np
import os

camera = cv.VideoCapture(0)
# Pegando o treinamento do cascade
car_cascade = cv.CascadeClassifier("treinamento/cascade.xml")

while True:
    # Iniciando a camera
    _, frame = camera.read()
    # Aplicando filtro hsv
    framehsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Retorno de array da cor verde
    menorverde = np.array([57,147,129])
    maiorverde = np.array([255,255,255])

    # Gera máscara de cor
    maskverde = cv.inRange(framehsv, menorverde, maiorverde)
    # Aplica máscara de cor
    resultadoverde = cv.bitwise_and(frame, frame, mask=maskverde)
    # Aplica camada cinza para processamento
    framecinzaverde = cv.cvtColor(resultadoverde, cv.COLOR_BGR2GRAY)
    # Gera thresh // limitre da cor
    _, threshverde = cv.threshold(framecinzaverde, 3, 255, cv.THRESH_BINARY)
    # Pegando partes da cor detectada na cam
    contornosverde, _ = cv.findContours(threshverde, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Gerando posições onde verde foi detectado
    for contorno in contornosverde:
        # Posições do contorno da cor
        (x, y, w, h) = cv.boundingRect(contorno)
        # Tamanho da area 
        area = cv.contourArea(contorno)
        
        if area > 1000:
            # Coloca texto na cam
            cv.putText(frame, "Verde detectado", (10, 50),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
            # Desenha contornos ao redor da detecção
            cv.drawContours(frame, contorno, -1, (0, 0, 0), 5)
            # Desenha um retângulo na area detectada
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # Recorta o ponto de interesse
            recorte = frame[y:y+h,x:x+w]
            cv.imshow("Recorte", recorte)

    cv.imshow("Imagem", frame)
    k = cv.waitKey(60)
    if k == 27:
        break

cv.destroyAllWindows()
camera.release
