import cv2
import os
from google.cloud import vision
import io

# --- CONFIGURAÇÃO ---
# Define o caminho para o seu arquivo de chave JSON
# O script espera que o arquivo 'chave-google.json' esteja na mesma pasta
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'chave-google.json'

# Nome do arquivo temporário para a foto da placa
NOME_ARQUIVO_FOTO = "placa_capturada.jpg"
# --------------------

def analisar_placa(caminho_imagem):
    """Envia uma imagem para a API do Google Vision e retorna o texto detectado."""
    print(f"\nEnviando '{caminho_imagem}' para análise no Google Cloud Vision...")

    try:
        # Inicializa o cliente da API
        client = vision.ImageAnnotatorClient()

        # Carrega a imagem em memória
        with io.open(caminho_imagem, 'rb') as image_file:
            content = image_file.read()

        image = vision.Image(content=content)

        # Realiza a detecção de texto
        response = client.text_detection(image=image)
        texts = response.text_annotations

        if response.error.message:
            raise Exception(
                '{}\nPara mais informações, verifique a documentação da API.'.format(
                    response.error.message))

        if texts:
            # A primeira detecção (texts[0]) geralmente é o texto completo encontrado
            placa_detectada = texts[0].description.strip().replace("\n", " ")
            print("--- TEXTO DETECTADO ---")
            print(placa_detectada)
            print("-----------------------")
        else:
            print("Nenhum texto foi detectado na imagem.")

    except Exception as e:
        print(f"ERRO: Ocorreu um problema ao chamar a API do Google Vision.")
        print(e)


# Inicializa a webcam (geralmente a câmera 0 é a padrão)
cap = cv2.VideoCapture(0)

print(">>> Câmera iniciada. Olhando para o vídeo...")
print(">>> Pressione 's' para salvar a foto e analisar a placa.")
print(">>> Pressione 'q' para sair.")

while True:
    # Captura frame a frame
    ret, frame = cap.read()

    if not ret:
        print("Erro ao capturar imagem da câmera.")
        break

    # Mostra o resultado em uma janela
    cv2.imshow('Protótipo Local - Pressione "s" para salvar ou "q" para sair', frame)

    # Espera por uma tecla
    key = cv2.waitKey(1) & 0xFF

    # Se a tecla 's' for pressionada, salva a imagem e a analisa
    if key == ord('s'):
        # Salva o frame atual como uma imagem
        cv2.imwrite(NOME_ARQUIVO_FOTO, frame)
        print(f"\nFoto salva como '{NOME_ARQUIVO_FOTO}'!")
        
        # Chama a função para analisar a imagem salva
        analisar_placa(NOME_ARQUIVO_FOTO)
        print("\n>>> Câmera pronta para a próxima captura. Pressione 's' ou 'q'.")

    # Se a tecla 'q' for pressionada, sai do loop
    elif key == ord('q'):
        print("Encerrando o programa.")
        break

# Quando tudo estiver feito, libera a captura e fecha as janelas
cap.release()
cv2.destroyAllWindows()