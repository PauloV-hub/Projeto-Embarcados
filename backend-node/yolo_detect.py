import sys
import argparse
import os
import cv2
import numpy as np
from ultralytics import YOLO
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

# Inicializa o FaceNet
embedder = FaceNet()

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

# Carrega modelos
detector = YOLO("yolov8n-face-lindevs.pt")
meus_embeddings = np.load('meus_embeddings.npy')  # Array de embeddings (n, 512)

# Calcula o embedding médio de referência
embedding_ref = np.mean(meus_embeddings, axis=0)
# Normaliza o embedding de referência
embedding_ref = embedding_ref / np.linalg.norm(embedding_ref)

def preprocess_face(face):
    """Prepara a face para o FaceNet"""
    face = cv2.resize(face, (160, 160))  # Tamanho esperado pelo FaceNet
    face = face.astype('float32')
    mean = face.mean()
    std = face.std()
    face = (face - mean) / std
    return np.expand_dims(face, axis=0)

def normalize_embedding(embedding):
    """Normaliza o embedding para comparação de similaridade"""
    return embedding / np.linalg.norm(embedding)

img = cv2.imread(args.source)
if img is None:
    print("ERROR", file=sys.stderr)
    sys.exit(1)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = detector(img_rgb)

max_similarity = 0.0

for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face = img_rgb[y1:y2, x1:x2]
        
        if face.size == 0:
            continue

        try:
            # Pré-processamento e extração de embedding
            face_processed = preprocess_face(face)
            embedding = embedder.embeddings(face_processed)[0]
            embedding = normalize_embedding(embedding)
            
            # Calcula similaridade com o embedding médio de referência
            similarity = cosine_similarity([embedding], [embedding_ref])[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
            
            # Ajuste o limiar conforme necessário (FaceNet geralmente requer >0.6 para match)
            color = (0, 255, 0) if similarity > 0.65 else (0, 0, 255)
            label = f"Você {similarity:.2f}" if similarity > 0.65 else f"Outro {similarity:.2f}"
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        except Exception as e:
            print(f"Erro no processamento: {str(e)}", file=sys.stderr)
            continue

# Salva imagem
os.makedirs(args.output, exist_ok=True)
output_path = os.path.join(args.output, os.path.basename(args.source))
cv2.imwrite(output_path, img)

print(f"{output_path}|{max_similarity:.4f}")