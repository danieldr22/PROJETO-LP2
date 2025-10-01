import cv2
import os
from google.cloud import vision
import io

# --- CONFIGURAÇÃO ---
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'chave-google.json'
NOME_ARQUIVO_FOTO = "foto_capturada.jpg"
# --------------------

def detectar_e_desenhar(caminho_imagem):
    """Detecta objetos, desenha na imagem e a exibe."""
    print(f"\nAnalisando '{caminho_imagem}' para detectar objetos...")

    try:
        client = vision.ImageAnnotatorClient()

        with io.open(caminho_imagem, 'rb') as image_file:
            content = image_file.read()

        image_for_api = vision.Image(content=content)

        # Chama a API para detectar objetos
        response = client.object_localization(image=image_for_api)
        objects = response.localized_object_annotations

        if response.error.message:
            raise Exception(f'{response.error.message}\nVerifique a documentação da API.')

        # Carrega a imagem com OpenCV para podermos desenhar nela
        image_to_draw = cv2.imread(caminho_imagem)
        height, width, _ = image_to_draw.shape
        
        contador_pessoas = 0

        if objects:
            print("--- OBJETOS DETECTADOS ---")
            for obj in objects:
                # O nome do objeto e a confiança da detecção
                nome_objeto = obj.name
                confianca = obj.score
                print(f"- Objeto: {nome_objeto}, Confiança: {confianca:.2%}")
                
                if nome_objeto == 'Person':
                    contador_pessoas += 1

                # As coordenadas do retângulo vêm normalizadas (de 0 a 1)
                # Precisamos convertê-las para pixels da imagem real
                vertices = obj.bounding_poly.normalized_vertices
                
                # Ponto inicial (canto superior esquerdo)
                pt1 = (int(vertices[0].x * width), int(vertices[0].y * height))
                # Ponto final (canto inferior direito)
                pt2 = (int(vertices[2].x * width), int(vertices[2].y * height))

                # --- DESENHANDO NA IMAGEM ---
                # Desenha o retângulo verde
                cv2.rectangle(image_to_draw, pt1, pt2, (0, 255, 0), 2)

                # Prepara o texto para exibir (nome e confiança)
                label = f"{nome_objeto} ({confianca:.0%})"
                # Posição do texto (um pouco acima do retângulo)
                label_position = (pt1[0], pt1[1] - 10)
                
                # Escreve o texto na imagem
                cv2.putText(image_to_draw, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            print("--------------------------")
            print(f"RESULTADO: {contador_pessoas} pessoa(s) detectada(s).")
            # Mostra a imagem final com os desenhos em uma nova janela
            cv2.imshow('Resultado da Deteccao', image_to_draw)
        else:
            print("RESULTADO: Nenhum objeto foi detectado na imagem.")

    except Exception as e:
        print(f"ERRO: {e}")


# --- LOOP PRINCIPAL DA CÂMERA (sem alterações) ---
cap = cv2.VideoCapture(0)

print(">>> Câmera iniciada. Olhando para o vídeo...")
print(">>> Pressione 's' para capturar e analisar a foto.")
print(">>> Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar imagem da câmera.")
        break

    cv2.imshow('Protótipo - Pressione "s" ou "q"', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        cv2.imwrite(NOME_ARQUIVO_FOTO, frame)
        print(f"\nFoto salva como '{NOME_ARQUIVO_FOTO}'!")
        
        # Chama a nova função que detecta e desenha
        detectar_e_desenhar(NOME_ARQUIVO_FOTO)
        print("\n>>> Câmera pronta para a próxima captura. Pressione 's' ou 'q'.")

    elif key == ord('q'):
        print("Encerrando o programa.")
        break

cap.release()
cv2.destroyAllWindows()
