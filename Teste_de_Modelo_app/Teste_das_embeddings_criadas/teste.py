# import cv2
# import numpy as np
# from ultralytics import YOLO
# from keras_facenet import FaceNet
# from sklearn.metrics.pairwise import cosine_similarity

# # Caminho da imagem que você quer testar
# caminho_imagem = "teste4.jpg"  # <- coloque aqui o nome da sua imagem

# # Carrega modelos
# detector = YOLO('yolov8n-face-lindevs.pt')  # ou 'best.pt' se for o modelo treinado por você
# embedder = FaceNet()

# # Carrega seu embedding salvo
# meu_embedding = np.load('meu_embedding_melhorado_2.npy')

# # Lê e converte a imagem
# img = cv2.imread(caminho_imagem)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # Detecta rostos
# resultados = detector(img_rgb)

# # Para cada rosto detectado
# for r in resultados:
#     for box in r.boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         rosto = img_rgb[y1:y2, x1:x2]

#         if rosto.size == 0:
#             continue

#         # Extrai embedding do rosto
#         embedding = embedder.embeddings([rosto])[0]

#         # Compara com seu embedding
#         similaridade = cosine_similarity([embedding], [meu_embedding])[0][0]

#         # Decide se é você
#         limiar = 0.8
#         if similaridade > limiar:
#             print(f"✅ É você! Similaridade: {similaridade:.4f}")
#             cor = (0, 255, 0)
#             texto = "Paulo Victor"
#         else:
#             print(f"❌ Não é você. Similaridade: {similaridade:.4f}")
#             cor = (0, 0, 255)
#             texto = "Outro"

#         # Desenha resultado
#         cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
#         cv2.putText(img, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

# cv2.imwrite("resultado.jpg", img)
# print("Imagem salva como resultado.jpg")

# cv2.destroyAllWindows()
import cv2
import numpy as np
from ultralytics import YOLO
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

def normalize_embedding(embedding):
    """Normaliza o embedding para comparação de similaridade"""
    return embedding / np.linalg.norm(embedding)

# Configurações
caminho_imagem = "teste1.jpg"  # Imagem para testar
embedding_file = "meus_embeddings.npy"  # Arquivo com múltiplos embeddings
limiar = 0.7  # Limiar ajustado para FaceNet

# Carrega modelos
detector = YOLO('yolov8n-face-lindevs.pt')
embedder = FaceNet()

# Carrega e processa os embeddings de referência
meus_embeddings = np.load(embedding_file)  # Array (n, 512)

# Calcula o embedding médio e normaliza
embedding_ref = np.mean(meus_embeddings, axis=0)
embedding_ref = normalize_embedding(embedding_ref)

# Lê e converte a imagem
img = cv2.imread(caminho_imagem)
if img is None:
    print("Erro: Não foi possível carregar a imagem")
    exit()

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Detecta rostos
resultados = detector(img_rgb)

# Para cada rosto detectado
for r in resultados:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        rosto = img_rgb[y1:y2, x1:x2]

        if rosto.size == 0:
            continue

        try:
            # Extrai e normaliza embedding do rosto detectado
            embedding = embedder.embeddings([rosto])[0]
            embedding = normalize_embedding(embedding)

            # Calcula similaridade com o embedding de referência
            similaridade = cosine_similarity([embedding], [embedding_ref])[0][0]

            # Exibe informações detalhadas
            print(f"Similaridade calculada: {similaridade:.4f}")
            print(f"Distância média entre embeddings: {np.mean(np.linalg.norm(meus_embeddings - embedding, axis=1)):.4f}")

            # Decisão com base no limiar
            if similaridade > limiar:
                print(f"✅ É você! Similaridade: {similaridade:.4f}")
                cor = (0, 255, 0)  # Verde
                texto = f"Victor {similaridade:.2f}"
            else:
                print(f"❌ Não é você. Similaridade: {similaridade:.4f}")
                cor = (0, 0, 255)  # Vermelho
                texto = f"Outro {similaridade:.2f}"

            # Desenha resultado na imagem
            cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(img, texto, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

        except Exception as e:
            print(f"Erro no processamento: {str(e)}")
            continue

# Salva e mostra o resultado
cv2.imwrite("resultado_com_media.jpg", img)
print("Resultado salvo como 'resultado_com_media.jpg'")

# Mostra a imagem (opcional - requer ambiente com suporte a GUI)
# cv2.imshow("Resultado", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()