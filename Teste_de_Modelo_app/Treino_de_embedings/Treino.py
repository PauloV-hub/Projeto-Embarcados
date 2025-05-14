import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from ultralytics import YOLO

# Configurações
FACE_PHOTOS_DIR = "minhas_fotos"  # Pasta com suas 10 fotos
OUTPUT_EMBEDDING = "meu_embedding_saiunesse.npy"  # Arquivo de saída
DETECTOR_MODEL = "yolov8n-face-lindevs.pt"  # Modelo de detecção facial

# Inicializa modelos
detector = YOLO(DETECTOR_MODEL)
embedder = FaceNet()

def preprocess_face(face_img):
    """Prepara o rosto para o FaceNet"""
    face_img = cv2.resize(face_img, (160, 160))
    face_img = face_img.astype('float32')
    mean = face_img.mean()
    std = face_img.std()
    return (face_img - mean) / std

def get_face_embedding(face_img):
    """Extrai embedding de uma face"""
    face_processed = preprocess_face(face_img)
    return embedder.embeddings(np.expand_dims(face_processed, axis=0))[0]

# Lista para armazenar todos os embeddings
all_embeddings = []

# Processa cada imagem na pasta
for filename in os.listdir(FACE_PHOTOS_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(FACE_PHOTOS_DIR, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detecta rostos
        results = detector(img_rgb)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = img_rgb[y1:y2, x1:x2]
                
                if face.size == 0:
                    continue
                
                try:
                    embedding = get_face_embedding(face)
                    all_embeddings.append(embedding)
                    print(f"Processada: {filename} | Embeddings coletados: {len(all_embeddings)}")
                except Exception as e:
                    print(f"Erro em {filename}: {str(e)}")

# Calcula o embedding médio
if len(all_embeddings) > 0:
    all_embeddings_np = np.array(all_embeddings)
    np.save("meus_embeddings.npy", all_embeddings_np)
    print(f"\n{len(all_embeddings)} embeddings salvos em 'meus_embeddings.npy'")
    print(f"Baseado em {len(all_embeddings)} imagens processadas")
else:
    print("Nenhum embedding foi gerado. Verifique suas imagens.")