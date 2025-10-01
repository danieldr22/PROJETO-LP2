import cv2
import os
from google.cloud import vision
import io

# --- CONFIGURAÇÃO ---
# O script usa a mesma chave que o protótipo anterior
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'chave-google.json'

# Nome do arquivo temporário para a foto
NOME_ARQUIVO_FOTO = "foto_capturada.jpg"
# --------------------

def detectar_pessoas(caminho_imagem):
    """Envia uma imagem para a API do Google Vision e conta quantas pessoas foram detectadas."""
    print(f"\nAnalisando '{caminho_imagem}' para detectar pessoas...")

    try:
        # Inicializa o cliente da API
        client = vision.ImageAnnotatorClient()

        # Carrega a imagem em memória
        with io.open(caminho_imagem, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # MUDANÇA AQUI: Usamos a detecção de objetos em vez da detecção de texto
        response = client.object_localization(image=image)
        objects = response.localized_object_annotations

        if response.error.message:
            raise Exception(
                '{}\nPara mais informações, verifique a documentação da API.'.format(
                    response.error.message))

        # Contagem de pessoas
        contador_pessoas = 0
        if objects:
            print("--- OBJETOS DETECTADOS ---")
            for obj in objects:
                # O nome do objeto detectado fica em 'obj.name'
                print(f"- Objeto: {obj.name}, Confiança: {obj.score:.2%}")
                if obj.name == 'Person':
                    contador_pessoas += 1
            print("--------------------------")
        
        if contador_pessoas > 0:
            print(f"RESULTADO: {contador_pessoas} pessoa(s) detectada(s) na imagem.")
        else:
            print("RESULTADO: Nenhuma pessoa foi detectada na imagem.")

    except Exception as e:
        print(f"ERRO: Ocorreu um problema ao chamar a API do Google Vision.")
        print(e)


# Inicializa a webcam
cap = cv2.VideoCapture(0)

print(">>> Câmera iniciada. Olhando para o vídeo...")
print(">>> Pressione 's' para capturar a foto e detectar pessoas.")
print(">>> Pressione 'q' para sair.")

while True:
    # Captura frame a frame
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar imagem da câmera.")
        break

    # Mostra o resultado em uma janela
    cv2.imshow('Protótipo Pessoas - Pressione "s" para capturar ou "q" para sair', frame)

    # Espera por uma tecla
    key = cv2.waitKey(1) & 0xFF

    # Se a tecla 's' for pressionada, salva a imagem e a analisa
    if key == ord('s'):
        cv2.imwrite(NOME_ARQUIVO_FOTO, frame)
        print(f"\nFoto salva como '{NOME_ARQUIVO_FOTO}'!")
        
        # Chama a nova função para detectar pessoas
        detectar_pessoas(NOME_ARQUIVO_FOTO)
        print("\n>>> Câmera pronta para a próxima captura. Pressione 's' ou 'q'.")

    # Se a tecla 'q' for pressionada, sai do loop
    elif key == ord('q'):
        print("Encerrando o programa.")
        break

# Libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()
