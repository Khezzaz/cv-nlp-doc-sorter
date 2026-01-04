"""
Test de l'endpoint /ocr/extract_with_boxes
Extraction de texte avec coordonnées des zones
"""

import requests
import json
from pathlib import Path
import cv2
import numpy as np

# Configuration
BASE_URL = "http://localhost:8000"
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

def test_extract_with_boxes():
    """Test de l'extraction avec bounding boxes"""
    
    print("\n" + "="*60)
    print("TEST: Extraction OCR avec Bounding Boxes")
    print("="*60 + "\n")
    
    # Chercher un document
    image_files = list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.png"))
    image_files = [f for f in image_files if 'generated' not in f.name.lower()]
    
    if not image_files:
        print("❌ Aucun document trouvé dans data/")
        return
    
    test_file = image_files[0]
    print(f"📄 Document testé: {test_file.name}\n")
    
    # Appeler l'API
    with open(test_file, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            f"{BASE_URL}/ocr/extract_with_boxes",
            files=files,
            timeout=30
        )
    
    if response.status_code != 200:
        print(f"❌ Erreur {response.status_code}: {response.text}")
        return
    
    result = response.json()
    
    # Afficher les résultats
    print(f"✅ Extraction réussie")
    print(f"   Nombre de blocs de texte: {result['total_blocks']}")
    print(f"   Dimensions image: {result['image_shape']['width']} x {result['image_shape']['height']}")
    print(f"\n{'='*60}")
    print("BLOCS DE TEXTE DÉTECTÉS")
    print("="*60 + "\n")
    
    for block in result['text_blocks']:
        print(f"📍 Bloc {block['id'] + 1}:")
        print(f"   Texte: \"{block['text']}\"")
        print(f"   Confiance: {block['confidence']:.1%}")
        print(f"   Position:")
        print(f"     - xmin: {block['bbox']['xmin']}")
        print(f"     - ymin: {block['bbox']['ymin']}")
        print(f"     - xmax: {block['bbox']['xmax']}")
        print(f"     - ymax: {block['bbox']['ymax']}")
        print(f"     - Largeur: {block['bbox']['xmax'] - block['bbox']['xmin']} px")
        print(f"     - Hauteur: {block['bbox']['ymax'] - block['bbox']['ymin']} px")
        print()
    
    # Sauvegarder le résultat
    OUTPUT_DIR.mkdir(exist_ok=True)
    output_file = OUTPUT_DIR / f"boxes_{test_file.stem}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Résultat sauvegardé: {output_file}")
    
    # Créer une visualisation
    visualize_boxes(test_file, result)

def visualize_boxes(image_path, result):
    """Créer une image avec les bounding boxes dessinées"""
    
    print("\n" + "="*60)
    print("VISUALISATION DES ZONES DE TEXTE")
    print("="*60 + "\n")
    
    # Charger l'image
    image = cv2.imread(str(image_path))
    
    if image is None:
        print("❌ Impossible de charger l'image")
        return
    
    # Créer une copie pour dessiner
    output_image = image.copy()
    
    # Couleurs pour les boxes (BGR)
    colors = [
        (0, 255, 0),    # Vert
        (255, 0, 0),    # Bleu
        (0, 0, 255),    # Rouge
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
    ]
    
    # Dessiner chaque bounding box
    for idx, block in enumerate(result['text_blocks']):
        bbox = block['bbox']
        color = colors[idx % len(colors)]
        
        # Rectangle
        cv2.rectangle(
            output_image,
            (bbox['xmin'], bbox['ymin']),
            (bbox['xmax'], bbox['ymax']),
            color,
            2
        )
        
        # Numéro du bloc
        cv2.putText(
            output_image,
            f"#{idx + 1}",
            (bbox['xmin'], bbox['ymin'] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )
    
    # Sauvegarder l'image avec les boxes
    output_path = OUTPUT_DIR / f"visualisation_{image_path.stem}.jpg"
    cv2.imwrite(str(output_path), output_image)
    
    print(f"✅ Visualisation créée: {output_path}")
    print(f"   Ouvrez ce fichier pour voir les zones de texte détectées")
    print(f"   {result['total_blocks']} zones sont encadrées et numérotées\n")

def create_zone_report(result, output_file):
    """Créer un rapport détaillé des zones"""
    
    report = []
    report.append("RAPPORT DES ZONES DE TEXTE DÉTECTÉES")
    report.append("="*60)
    report.append(f"\nDocument analysé: {result.get('filename', 'N/A')}")
    report.append(f"Nombre total de zones: {result['total_blocks']}")
    report.append(f"Dimensions image: {result['image_shape']['width']} x {result['image_shape']['height']}")
    report.append("\n" + "="*60)
    report.append("DÉTAILS PAR ZONE")
    report.append("="*60 + "\n")
    
    for block in result['text_blocks']:
        report.append(f"Zone #{block['id'] + 1}")
        report.append("-" * 40)
        report.append(f"Texte détecté: {block['text']}")
        report.append(f"Confiance OCR: {block['confidence']:.2%}")
        report.append(f"")
        report.append(f"Coordonnées de la zone:")
        report.append(f"  Position X: de {block['bbox']['xmin']} à {block['bbox']['xmax']}")
        report.append(f"  Position Y: de {block['bbox']['ymin']} à {block['bbox']['ymax']}")
        report.append(f"  Largeur: {block['bbox']['xmax'] - block['bbox']['xmin']} pixels")
        report.append(f"  Hauteur: {block['bbox']['ymax'] - block['bbox']['ymin']} pixels")
        report.append(f"")
        report.append(f"Coordonnées polygone complet:")
        for i, point in enumerate(block['polygon']):
            report.append(f"  Point {i+1}: x={point[0]}, y={point[1]}")
        report.append("\n")
    
    # Sauvegarder le rapport
    report_path = OUTPUT_DIR / f"rapport_zones_{output_file.stem}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
    
    print(f"📄 Rapport détaillé: {report_path}\n")

def main():
    print("\n" + "="*60)
    print("TEST DE L'ENDPOINT AVEC BOUNDING BOXES")
    print("Module OCR NLP - HAYTOM Manal")
    print("="*60)
    
    # Vérifier que l'API est lancée
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("\n✅ API opérationnelle")
        else:
            print("\n❌ API non accessible")
            return
    except:
        print("\n❌ Erreur: L'API n'est pas lancée")
        print("   Lancez d'abord: python main.py")
        return
    
    # Lancer le test
    test_extract_with_boxes()
    
    print("\n" + "="*60)
    print("UTILISATION DANS VOTRE CODE")
    print("="*60)
    print("""
Pour utiliser cet endpoint dans votre code:

import requests

response = requests.post(
    "http://localhost:8000/ocr/extract_with_boxes",
    files={"file": open("document.jpg", "rb")}
)

result = response.json()

# Parcourir les blocs de texte
for block in result["text_blocks"]:
    texte = block["text"]
    xmin = block["bbox"]["xmin"]
    ymin = block["bbox"]["ymin"]
    xmax = block["bbox"]["xmax"]
    ymax = block["bbox"]["ymax"]
    
    print(f"Texte: {texte} à la position ({xmin}, {ymin})")
""")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()