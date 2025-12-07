import pandas as pd
import io
from PIL import Image
import os

# --- CONFIGURATION ---
# Le nom exact de votre fichier parquet
parquet_file = "train-00000-of-00001-b64601da56687a05.parquet"
# Le dossier de sortie
output_dir = "dataset_extracted"
# ---------------------

# Créer le dossier de sortie
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print(f"Lecture du fichier {parquet_file}...")
# Lire le fichier Parquet
df = pd.read_parquet(parquet_file)

print(f"Extraction de {len(df)} images...")

for index, row in df.iterrows():
    try:
        # 1. Gérer l'image
        # Dans les fichiers Parquet HuggingFace, l'image est souvent un dictionnaire avec des bytes
        image_data = row['image']
        
        if isinstance(image_data, dict) and 'bytes' in image_data:
            image_bytes = image_data['bytes']
            image = Image.open(io.BytesIO(image_bytes))
        elif isinstance(image_data, bytes):
            image = Image.open(io.BytesIO(image_data))
        else:
            print(f"Format image inconnu ligne {index}")
            continue

        # 2. Gérer le texte (caption)
        # Le nom de la colonne peut changer, on vérifie les noms courants
        caption = ""
        if 'text' in row:
            caption = row['text']
        elif 'caption' in row:
            caption = row['caption']
        elif 'prompt' in row:
            caption = row['prompt']
        
        # 3. Sauvegarder
        filename = f"logo_{index:04d}"
        
        # Sauvegarder l'image en PNG
        image.save(f"{output_dir}/{filename}.png")
        
        # Sauvegarder le texte
        with open(f"{output_dir}/{filename}.txt", "w", encoding="utf-8") as f:
            f.write(str(caption))
            
    except Exception as e:
        print(f"Erreur à l'index {index}: {e}")

print(f"✅ Terminé ! Vos images sont dans le dossier '{output_dir}'")