# Bibliotecas importadas
import cv2 as cv 
import numpy as np
import face_recognition
import glob
import os
from datetime import datetime

# Gera lista para armazenar rostos
faces_encodings = []
# Gera lista para armazenar nome de cada rosto
faces_names = []
# Obtém o diretório atual
currentDir = os.getcwd()
# Abre a pasta faces
path = os.path.join(currentDir, "faces/")

# Lista os arquivos da pasta faces
lista = [f for f in glob.glob(path+"*.jpg")]
# Guarda o "tamanho" da lista, ou seja a qntd de rostos na pasta
tamanhoLista = len(lista)
# Captura o nome dado para cada rosto em uma variável
names = lista.copy()

# Gera laço de repetição para guardar singularidades de cada face
for i in range(tamanhoLista):
    # Carrega imagem por imagem 
    presets = face_recognition.load_image_file(lista[i])
    # Codifica cada face para comparar
    encoding = face_recognition.face_encodings(presets)[0]
    faces_encodings.append(encoding)
    names[i] = names[i].replace(currentDir, "")
    names[i] = names[i].replace(".jpg", "")
    names[i] = names[i].replace("faces", "")
    faces_names.append(names[i])

# Listas para trabalhar com os arquivos recebidos
face_locations = []
face_encodings = []
face_names = []

camera = cv.VideoCapture(0)
# Pegando o treinamento do cascade
car_cascade = cv.CascadeClassifier("treinamento/cascade.xml")

while True:
    # Chaves para abrir o cofre
    rostodetectado = False
    cordetectada = False
    maodetectada = False

    _, frame = camera.read()
    # Inverte a imagem da câmera
    frame = cv.flip(frame, 1)
    # Aplicando filtro hsv
    framehsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Parâmetros da cor verde
    menorverde = np.array([57,147,129])
    maiorverde = np.array([255,255,255])

    # Gera máscara de cor com os parâmetros já setados
    maskverde = cv.inRange(framehsv, menorverde, maiorverde)
    # Aplica máscara de cor
    resultadoverde = cv.bitwise_and(frame, frame, mask=maskverde)
    # Aplica camada cinza para processamento
    framecinzaverde = cv.cvtColor(resultadoverde, cv.COLOR_BGR2GRAY)
    # Gera thresh // limite da cor
    _, threshverde = cv.threshold(framecinzaverde, 3, 255, cv.THRESH_BINARY)
    # Pegando partes da cor detectada na cam
    contornosverde, _ = cv.findContours(threshverde, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Gerando posições onde verde foi detectado
    for contorno in contornosverde:
        # Posições do contorno da cor
        (x, y, w, h) = cv.boundingRect(contorno)
        # Tamanho da area 
        area = cv.contourArea(contorno)
        # Impede objetos muito pequenos de serem captados
        if area > 1000:
            cordetectada = True
            # Coloca texto na imagem mostrada
            cv.putText(frame, "Cor alvo detectada", (10, 50),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
            # Desenha contornos ao redor da detecção
            cv.drawContours(frame, contorno, -1, (0, 0, 0), 5)
            # Desenha um retângulo na area detectada
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
            # Recorta o ponto de interesse
            recorte = frame[y:y+h,x:x+w]
            cv.imshow("Recorte", recorte)

    # Aplicando filtro cinza
    framegray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Aplicando o Cascade
    mao = car_cascade.detectMultiScale(framegray, 1.2, 5)
     # Analise do cascade
    for (x,y,w,h) in mao:
        maodetectada = True
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    # Capturas de frame para identificação facial
    smallFrame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Captura de cores nos frames para singularidades dos rostos
    rgbSmallFrame = smallFrame[:, :, ::-1]
    # Acessa localização das fotos salvas para comparação com a imagem exibida
    face_locations = face_recognition.face_locations(rgbSmallFrame)
    face_encodings = face_recognition.face_encodings(
        rgbSmallFrame, face_locations)
    # Pega o nome do rosto detectado
    face_names = []

    # Gera laço para analisar se há um rosto "familiar"
    for face in face_encodings:
        # Faz a comparação da camera com o dataset
        matches = face_recognition.compare_faces(faces_encodings, face)
        # Define nome default como Desconhecido
        name = "Desconhecido"
        # Distância euclidiana para comparação das faces
        face_distances = face_recognition.face_distance(faces_encodings, face)
        # Retorna os índices dos valores mínimos ao longo do eixo euclidiano
        bestMatch = np.argmin(face_distances)
        # Se as imagens "baterem" altera o nome default para o nome da imagem
        if matches[bestMatch]:
            name = faces_names[bestMatch]
        # Adiciona o "nome da face" a lista de nomes
        face_names.append(name)

    # Faz a detecção do rosto e gera as posições
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top = top * 4
        right = right * 4
        bottom = bottom * 4
        left = left * 4
        rostodetectado = True
        # Desenha o contorno do rosto localizado
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
        cv.rectangle(frame, (left, bottom-35), (right, bottom), (0,0,255), cv.FILLED)
        cv.putText(frame, name, (left+4, bottom-4),
                    cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1)

    # Capta a data e hora atual
    data_e_hora_atuais = datetime.now()
    # Formata a escrita da data e pega a hora
    data_e_hora_em_texto = data_e_hora_atuais.strftime('%d/%m/%Y %H:%M')
    # Caso todas as chaves sejam apresentadas
    if rostodetectado and cordetectada and maodetectada == True:
        # Texto para abrir o cofre
        cv.putText(frame, "Cofre desbloqueado!", (10, 100),
            cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
        # Gera log de quem abriu
        try:
            # Abre o arquivo e escreve
            arquivo = open("abriu_log/quem_quando.txt", 'a')
            arquivo.writelines(data_e_hora_em_texto+" "+name+"\n")
        except:
            # Confere se existe o arquivo, caso contrário cria e escreve
            arquivo = open("abriu_log/quem_quando.txt", 'w+')
            arquivo.writelines(data_e_hora_em_texto+" "+name+"\n")
        # Fecha o arquivo log
        arquivo.close()

    cv.imshow("Imagem", frame)
    k = cv.waitKey(30)
    if k == 27:
        break

cv.destroyAllWindows()
camera.release
