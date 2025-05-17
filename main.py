# · Paso 0: Preparación de Librerías necesarias, relacionadas con la IA YOLO (You Only Look Once), para el proyecto
# pip install torch torchvision torchaudio 
# pip install opencv-python numpy 
# pip install ultralytics


# · Paso 1: Código principal
from ultralytics import YOLO
import cv2

# Cargamos el último modelo YOLO preentrenado (a su vez, la versión más potente: 'x', aunque se usará la 's' debido a resultados/rendimiento)
modelo = YOLO('yolo11s.pt')

# Cargamos el video objeto de análisis (también se puede capturar una cámara en tiempo real: 'cv2.VideoCapture(0)')
video = cv2.VideoCapture("../Videos/paseo_crowd.mp4")

while True:
    ret, frame = video.read()
    if not ret:
        break

    resultados = modelo(frame)

    # Convertimos los resultados a una lista de objetos detectados para un manejo más sencillo
    detecciones = resultados[0].boxes.data.cpu().numpy()

    contadorPersonas = 0

    # Dibujamos cajas y contamos las respectivas personas
    for *box, conf, cls in detecciones:
        id_clase = int(cls)
        if id_clase == 0:  # clase 0 = persona
            contadorPersonas += 1
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Persona", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Finalmente, se muestra el contador en pantalla
    cv2.putText(frame, f'Personas detectadas: {contadorPersonas}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Contador de Personas', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Presiona ESC para salir
        break

video.release()
cv2.destroyAllWindows()