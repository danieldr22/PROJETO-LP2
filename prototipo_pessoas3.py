import cv2
import os
import threading
import time
from google.cloud import vision

# --- CONFIGURAÇÃO ---
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'chave-google.json'
# Intervalo em segundos entre cada análise da API. 1.5 = 1 análise a cada 1.5 segundos.
# Diminua para análises mais rápidas (e maior custo), aumente para o contrário.
INTERVALO_ANALISE = 1.5
# --------------------

# --- Variáveis Globais para comunicação entre as threads ---
# Armazena os últimos objetos detectados pela API
latest_objects = []
# Lock para garantir que não haja conflito ao acessar a variável 'latest_objects'
lock = threading.Lock()
# Frame mais recente a ser analisado pela thread de fundo
frame_para_analisar = None
# Flag para encerrar a thread de fundo
parar_thread = False
# ---------------------------------------------------------

def thread_analise_google():
    """
    Esta função roda em uma thread separada.
    Ela pega o frame mais recente, envia para a API do Google e atualiza os resultados.
    """
    global latest_objects, frame_para_analisar

    client = vision.ImageAnnotatorClient()

    while not parar_thread:
        frame_local = None
        
        # Pega o frame mais recente para analisar
        with lock:
            if frame_para_analisar is not None:
                frame_local = frame_para_analisar.copy()
                frame_para_analisar = None # Limpa para não analisar o mesmo frame de novo

        if frame_local is not None:
            # Converte o frame do OpenCV para o formato de bytes que a API precisa
            success, encoded_image = cv2.imencode('.jpg', frame_local)
            if success:
                content = encoded_image.tobytes()
                image = vision.Image(content=content)

                # Chama a API
                try:
                    response = client.object_localization(image=image)
                    objects = response.localized_object_annotations
                    
                    # Atualiza a variável global com os novos resultados
                    with lock:
                        latest_objects = objects
                except Exception as e:
                    print(f"Erro na API do Google: {e}")
        
        # Espera um pouco antes de verificar por um novo frame
        time.sleep(0.1)

# --- LOOP PRINCIPAL DA CÂMERA ---

# Inicia e abre a webcam
cap = cv2.VideoCapture(0)
print(">>> Câmera iniciada.")

# Cria e inicia a thread de fundo que fará a análise
analysis_thread = threading.Thread(target=thread_analise_google, daemon=True)
analysis_thread.start()
print(">>> Thread de análise iniciada em segundo plano.")
print(">>> Pressione 'q' para sair.")

last_analysis_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Verifica se já passou tempo suficiente para disparar uma nova análise
    if time.time() - last_analysis_time > INTERVALO_ANALISE:
        with lock:
            # Disponibiliza o frame atual para a thread de análise
            frame_para_analisar = frame.copy()
        last_analysis_time = time.time()

    # Pega os resultados mais recentes para desenhar
    objetos_para_desenhar = []
    with lock:
        if latest_objects:
            objetos_para_desenhar = list(latest_objects)

    # Desenha os retângulos no frame ATUAL com base nos últimos resultados
    if objetos_para_desenhar:
        height, width, _ = frame.shape
        for obj in objetos_para_desenhar:
            label = f"{obj.name} ({obj.score:.0%})"
            vertices = obj.bounding_poly.normalized_vertices
            pt1 = (int(vertices[0].x * width), int(vertices[0].y * height))
            pt2 = (int(vertices[2].x * width), int(vertices[2].y * height))
            
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(frame, label, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Mostra o vídeo (com ou sem os desenhos)
    cv2.imshow('Analise em Tempo Real - Pressione "q" para sair', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        parar_thread = True # Sinaliza para a thread de fundo parar
        break

# Encerramento
print("Encerrando...")
analysis_thread.join(timeout=2) # Espera a thread terminar
cap.release()
cv2.destroyAllWindows()
