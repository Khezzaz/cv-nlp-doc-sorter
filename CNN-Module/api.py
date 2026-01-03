"""
api.py - Backend FastAPI pour la classification de documents CNN
Usage: uvicorn api:app --reload --port 8001
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
import uvicorn
import cv2
import numpy as np
import os
from datetime import datetime
from pathlib import Path

# Import du module CNN
from cnn_fewshot_classifier import (
    CNNFeatureExtractor, 
    FewShotDocumentClassifier,
    load_reference_images,
    ModelTester
)

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration centralisée de l'application"""
    MODEL_PATH = "models/cnn_fewshot.pkl"
    UPLOAD_DIR = "uploads"
    RESULTS_DIR = "results"
    REFERENCES_DIR = "references"
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.jfif'}

# =============================================================================
# MODÈLES PYDANTIC (Schémas de données)
# =============================================================================

class ClassificationResponse(BaseModel):
    """Schéma de réponse pour la classification d'un document"""
    predicted_class: str  # Ex: "CIN", "Facture", "Releve_Bancaire"
    confidence: float  # Ex: 0.95 (95%)
    scores: Dict[str, float]  # Ex: {"CIN": 0.95, "Facture": 0.03, "Releve_Bancaire": 0.02}
    low_confidence_warning: bool  # True si confidence < seuil
    neighbors: Optional[List[Dict]] = None  # Détails des voisins similaires (optionnel)
    avg_similarity: Optional[float] = None  # Similarité moyenne avec les voisins
    processing_time_ms: float  # Temps de traitement en millisecondes

class ModelStats(BaseModel):
    """Statistiques du modèle chargé"""
    n_references: int  # Nombre total d'images de référence
    class_distribution: Dict[str, int]  # Ex: {"CIN": 10, "Facture": 10, "RB": 10}
    last_updated: Optional[str]  # Date de dernière mise à jour
    model_loaded: bool  # True si le modèle est chargé
    backbone: str  # Ex: "resnet50"

class TrainingRequest(BaseModel):
    """Paramètres pour l'entraînement du modèle"""
    reference_path: str = Field(default="references", description="Chemin vers le dossier de références")
    k_neighbors: int = Field(default=3, ge=1, le=10, description="Nombre de voisins k-NN")
    confidence_threshold: float = Field(default=0.4, ge=0.0, le=1.0, description="Seuil de confiance")

class EvaluationRequest(BaseModel):
    """Paramètres pour l'évaluation du modèle"""
    test_directory: str = Field(..., description="Chemin vers le dossier de test")

class BatchResult(BaseModel):
    """Résultat pour une image dans un batch"""
    filename: str
    predicted_class: Optional[str] = None
    confidence: Optional[float] = None
    scores: Optional[Dict[str, float]] = None
    error: Optional[str] = None

class BatchClassificationResponse(BaseModel):
    """Réponse pour une classification en batch"""
    results: List[BatchResult]
    total_images: int
    successful: int
    failed: int
    processing_time_ms: float

# =============================================================================
# VARIABLES GLOBALES
# =============================================================================

# Ces variables seront initialisées au démarrage
classifier: Optional[FewShotDocumentClassifier] = None
extractor: Optional[CNNFeatureExtractor] = None

# =============================================================================
# LIFESPAN EVENTS - Gestion du cycle de vie de l'application
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestion du cycle de vie de l'application
    - Exécuté au DÉMARRAGE: Initialise le modèle
    - Exécuté à l'ARRÊT: Nettoie les ressources
    """
    global classifier, extractor
    
    # ========== STARTUP ==========
    print("\n" + "="*60)
    print("🚀 DÉMARRAGE DE L'API CNN")
    print("="*60)
    
    # Créer les dossiers nécessaires
    ensure_directories()
    
    try:
        # 1. Initialiser l'extracteur CNN (ResNet50)
        print("🔧 Initialisation du modèle ResNet50...")
        extractor = CNNFeatureExtractor(backbone='resnet50')
        
        # 2. Créer le classifier Few-Shot
        classifier = FewShotDocumentClassifier(
            extractor, 
            k=3,  # 3 voisins pour k-NN
            confidence_threshold=0.4  # Seuil de confiance 40%
        )
        
        # 3. Charger le modèle sauvegardé (si existe)
        if os.path.exists(Config.MODEL_PATH):
            print(f"📂 Chargement du modèle: {Config.MODEL_PATH}")
            classifier.load(Config.MODEL_PATH)
            print(f"✅ Modèle chargé: {classifier.stats['n_references']} références")
            print(f"📊 Distribution: {classifier.stats['class_distribution']}")
        else:
            print("⚠️  Aucun modèle trouvé. Utilisez POST /train pour entraîner")
        
        print("\n✅ API prête sur http://localhost:8001")
        print(f"📖 Documentation: http://localhost:8001/docs")
        print("="*60 + "\n")
    
    except Exception as e:
        print(f"❌ Erreur d'initialisation: {e}")
        classifier = None
    
    yield  # L'application tourne ici (entre startup et shutdown)
    
    # ========== SHUTDOWN ==========
    print("\n👋 Arrêt de l'API CNN...")

# =============================================================================
# APPLICATION FASTAPI
# =============================================================================

app = FastAPI(
    title="🖼️ CNN Document Classification API",
    description="API de classification de documents (CIN, Facture, Relevé Bancaire) basée sur CNN Few-Shot Learning",
    version="1.0.0",
    docs_url="/docs",  # Documentation Swagger
    redoc_url="/redoc",  # Documentation ReDoc
    lifespan=lifespan  # Gestionnaire de cycle de vie
)

# Configuration CORS (Cross-Origin Resource Sharing)
# Permet à d'autres domaines d'appeler l'API (ex: frontend React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # À restreindre en production (ex: ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],
)

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def ensure_directories():
    """Crée les dossiers nécessaires s'ils n'existent pas"""
    for directory in [Config.UPLOAD_DIR, Config.RESULTS_DIR, "models", Config.REFERENCES_DIR]:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Dossier vérifié: {directory}")

async def validate_image(file: UploadFile) -> bytes:
    """
    Valide une image uploadée
    - Vérifie l'extension
    - Vérifie la taille
    - Retourne le contenu en bytes
    """
    # Vérification de l'extension
    ext = Path(file.filename).suffix.lower()
    if ext not in Config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté. Formats acceptés: {', '.join(Config.ALLOWED_EXTENSIONS)}"
        )
    
    # Lecture du fichier
    contents = await file.read()
    
    # Vérification de la taille
    if len(contents) > Config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Fichier trop volumineux. Taille max: {Config.MAX_FILE_SIZE / 1024 / 1024:.1f}MB"
        )
    
    return contents

def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """
    Convertit des bytes en image OpenCV
    bytes → numpy array → image OpenCV (BGR)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)  # Conversion en array numpy
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Décodage en image BGR
    
    if img is None:
        raise HTTPException(status_code=400, detail="Image corrompue ou invalide")
    
    return img

# =============================================================================
# ROUTES - INFORMATIONS & SANTÉ
# =============================================================================

@app.get("/", tags=["Info"])
async def root():
    """
    Route racine - Informations sur l'API
    Accessible via: GET http://localhost:8001/
    """
    return {
        "name": "CNN Document Classification API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": classifier is not None and classifier.knn is not None,
        "endpoints": {
            "health": "GET /health",
            "stats": "GET /stats",
            "classes": "GET /classes",
            "classify": "POST /classify",
            "classify_batch": "POST /classify/batch",
            "train": "POST /train",
            "add_reference": "POST /train/add-reference",
            "evaluate": "POST /test/evaluate",
            "docs": "GET /docs"
        }
    }

@app.get("/health", tags=["Info"])
async def health_check():
    """
    Vérifie l'état de santé de l'API
    Utilisé par le Gateway pour vérifier que le service est opérationnel
    """
    model_loaded = classifier is not None and classifier.knn is not None
    
    return {
        "service": "cnn_classification",
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "n_references": classifier.stats.get('n_references', 0) if classifier else 0,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/stats", response_model=ModelStats, tags=["Info"])
async def get_stats():
    """
    Récupère les statistiques du modèle chargé
    - Nombre de références
    - Distribution des classes
    - Date de dernière mise à jour
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modèle non initialisé")
    
    return ModelStats(
        n_references=classifier.stats.get('n_references', 0),
        class_distribution=classifier.stats.get('class_distribution', {}),
        last_updated=classifier.stats.get('last_updated'),
        model_loaded=classifier.knn is not None,
        backbone=extractor.backbone if extractor else "unknown"
    )

@app.get("/classes", tags=["Info"])
async def get_classes():
    """
    Retourne la liste des classes supportées
    Ex: ["CIN", "Facture", "Releve_Bancaire"]
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Modèle non initialisé")
    
    return {
        "classes": classifier.class_names,
        "n_classes": len(classifier.class_names),
        "distribution": classifier.stats.get('class_distribution', {})
    }

# =============================================================================
# ROUTES - CLASSIFICATION (CŒUR DE L'API)
# =============================================================================

@app.post("/classify", response_model=ClassificationResponse, tags=["Classification"])
async def classify_image(
    file: UploadFile = File(..., description="Image du document à classifier"),
    return_details: bool = True
):
    """
    🎯 ROUTE PRINCIPALE: Classifie un document
    
    **Workflow:**
    1. Réception de l'image
    2. Validation (format, taille)
    3. Conversion en format OpenCV
    4. Prétraitement (débruitage, deskewing)
    5. Extraction des features (ResNet50)
    6. Classification (k-NN)
    7. Retour du résultat
    
    **Exemple d'utilisation:**
    ```python
    import requests
    
    files = {'file': open('document.jpg', 'rb')}
    response = requests.post('http://localhost:8001/classify', files=files)
    print(response.json())
    ```
    
    **Réponse attendue:**
    {
        "predicted_class": "CIN",
        "confidence": 0.95,
        "scores": {"CIN": 0.95, "Facture": 0.03, "Releve_Bancaire": 0.02},
        "low_confidence_warning": false,
        "processing_time_ms": 150.5
    }
    """
    
    # Vérification que le modèle est chargé
    if classifier is None or classifier.knn is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle non chargé. Utilisez POST /train pour entraîner le modèle."
        )
    
    start_time = datetime.now()
    
    try:
        # 1. Validation de l'image
        contents = await validate_image(file)
        
        # 2. Conversion en image OpenCV
        img = bytes_to_cv2(contents)
        
        # 3. Classification (prétraitement + extraction + k-NN)
        result = classifier.classify(img, return_details=return_details)
        
        # 4. Calcul du temps de traitement
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result['processing_time_ms'] = processing_time
        
        # 5. Retour du résultat
        return ClassificationResponse(**result)
    
    except HTTPException:
        raise  # Propager les erreurs HTTP
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de la classification: {str(e)}"
        )

@app.post("/classify/batch", response_model=BatchClassificationResponse, tags=["Classification"])
async def classify_batch(
    files: List[UploadFile] = File(..., description="Liste d'images à classifier"),
    return_details: bool = False
):
    """
    📦 Classification en batch (plusieurs images à la fois)
    
    **Avantages:**
    - Traitement parallélisé
    - Plus rapide que plusieurs appels individuels
    - Rapport global des résultats
    
    **Limites:**
    - Maximum 50 images par requête
    - Taille max par image: 10MB
    """
    if classifier is None or classifier.knn is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle non chargé. Utilisez POST /train d'abord."
        )
    
    if len(files) > 50:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 50 images par batch"
        )
    
    start_time = datetime.now()
    results = []
    successful = 0
    failed = 0
    
    # Traiter chaque image
    for file in files:
        try:
            contents = await validate_image(file)
            img = bytes_to_cv2(contents)
            
            result = classifier.classify(img, return_details=return_details)
            
            results.append(BatchResult(
                filename=file.filename,
                predicted_class=result['predicted_class'],
                confidence=result['confidence'],
                scores=result['scores']
            ))
            successful += 1
        
        except Exception as e:
            # En cas d'erreur sur une image, continuer avec les autres
            results.append(BatchResult(
                filename=file.filename,
                error=str(e)
            ))
            failed += 1
    
    processing_time = (datetime.now() - start_time).total_seconds() * 1000
    
    return BatchClassificationResponse(
        results=results,
        total_images=len(files),
        successful=successful,
        failed=failed,
        processing_time_ms=processing_time
    )

# =============================================================================
# ROUTES - ENTRAÎNEMENT
# =============================================================================

@app.post("/train", tags=["Training"])
async def train_model(config: TrainingRequest):
    """
    🎓 Entraîne le modèle avec des images de référence
    
    **Structure attendue du dossier références:**
    ```
    references/
    ├── CIN/
    │   ├── cin_001.jpg
    │   ├── cin_002.jpg
    │   └── ...
    ├── FACTURE/
    │   ├── facture_001.jpg
    │   ├── facture_002.jpg
    │   └── ...
    └── RB/
        ├── rb_001.jpg
        ├── rb_002.jpg
        └── ...
    ```
    
    **Processus:**
    1. Charger toutes les images du dossier
    2. Prétraiter chaque image
    3. Extraire les embeddings (ResNet50)
    4. Construire la base vectorielle
    5. Construire le modèle k-NN
    6. Sauvegarder le modèle
    
    **Exemple:**
    ```python
    import requests
    
    payload = {
        "reference_path": "references",
        "k_neighbors": 3,
        "confidence_threshold": 0.4
    }
    response = requests.post('http://localhost:8001/train', json=payload)
    print(response.json())
    ```
    """
    global classifier
    
    # Vérification du dossier
    if not os.path.exists(config.reference_path):
        raise HTTPException(
            status_code=404, 
            detail=f"Dossier de références introuvable: {config.reference_path}"
        )
    
    try:
        print(f"\n🎓 Début de l'entraînement...")
        print(f"📁 Dossier: {config.reference_path}")
        print(f"🔢 k_neighbors: {config.k_neighbors}")
        print(f"📊 confidence_threshold: {config.confidence_threshold}")
        
        # 1. Chargement des images de référence
        imgs, lbls, meta = load_reference_images(config.reference_path, verbose=True)
        
        if len(imgs) == 0:
            raise HTTPException(
                status_code=400, 
                detail="Aucune image de référence trouvée. Vérifiez la structure du dossier."
            )
        
        # 2. Réinitialisation du classifier avec nouveaux paramètres
        classifier = FewShotDocumentClassifier(
            extractor, 
            k=config.k_neighbors,
            confidence_threshold=config.confidence_threshold
        )
        
        # 3. Ajout des références (extraction des embeddings)
        classifier.add_references(imgs, lbls, meta, use_batch=True)
        
        # 4. Sauvegarde du modèle
        classifier.save(Config.MODEL_PATH)
        
        print(f"✅ Entraînement terminé!\n")
        
        return {
            "status": "success",
            "message": "Modèle entraîné avec succès",
            "n_references": len(imgs),
            "class_distribution": classifier.stats['class_distribution'],
            "model_path": Config.MODEL_PATH,
            "k_neighbors": config.k_neighbors,
            "confidence_threshold": config.confidence_threshold
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de l'entraînement: {str(e)}"
        )

@app.post("/train/add-reference", tags=["Training"])
async def add_reference(
    file: UploadFile = File(...),
    label: str = Form(..., description="Classe: CIN, Facture ou Releve_Bancaire")
):
    """
    ➕ Ajoute une nouvelle image de référence au modèle existant
    
    **Avantage:**
    - Pas besoin de tout ré-entraîner
    - Ajout incrémental d'exemples
    - Amélioration progressive du modèle
    
    **Exemple:**
    ```python
    import requests
    
    files = {'file': open('nouvelle_cin.jpg', 'rb')}
    data = {'label': 'CIN'}
    response = requests.post('http://localhost:8001/train/add-reference', 
                            files=files, data=data)
    print(response.json())
    ```
    """
    if classifier is None or classifier.knn is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle non chargé. Entraînez d'abord avec POST /train"
        )
    
    if label not in classifier.class_names:
        raise HTTPException(
            status_code=400, 
            detail=f"Classe invalide '{label}'. Classes valides: {', '.join(classifier.class_names)}"
        )
    
    try:
        contents = await validate_image(file)
        img = bytes_to_cv2(contents)
        
        # Sauvegarde de l'image pour traçabilité
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{label}_{timestamp}_{file.filename}"
        save_path = os.path.join(Config.UPLOAD_DIR, filename)
        cv2.imwrite(save_path, img)
        
        # Ajout au modèle (extraction embedding + mise à jour k-NN)
        classifier.add_references(
            [img], 
            [label], 
            [{"path": save_path, "filename": filename}]
        )
        
        # Sauvegarde du modèle mis à jour
        classifier.save(Config.MODEL_PATH)
        
        return {
            "status": "success",
            "message": "Référence ajoutée avec succès",
            "filename": filename,
            "label": label,
            "total_references": classifier.stats['n_references'],
            "class_distribution": classifier.stats['class_distribution']
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de l'ajout: {str(e)}"
        )

# =============================================================================
# ROUTES - ÉVALUATION & TESTS
# =============================================================================

@app.post("/test/evaluate", tags=["Testing"])
async def evaluate_model(request: EvaluationRequest):
    """
    🧪 Évalue les performances du modèle sur un jeu de test
    
    **Structure attendue:**
    ```
    test/
    ├── CIN/
    │   ├── test_cin_001.jpg
    │   └── ...
    ├── Facture/
    │   ├── test_facture_001.jpg
    │   └── ...
    └── Releve_Bancaire/
        ├── test_rb_001.jpg
        └── ...
    ```
    
    **Métriques calculées:**
    - Accuracy globale
    - Precision, Recall, F1-Score par classe
    - Matrice de confusion
    - Rapport détaillé au format JSON
    
    **Retour:**
    ```json
    {
        "status": "success",
        "accuracy": 90.0,
        "n_correct": 45,
        "n_total": 50,
        "class_stats": {
            "CIN": {"n_samples": 20, "n_correct": 18, "accuracy": 90.0},
            "Facture": {"n_samples": 20, "n_correct": 17, "accuracy": 85.0},
            "Releve_Bancaire": {"n_samples": 10, "n_correct": 10, "accuracy": 100.0}
        }
    }
    ```
    """
    if classifier is None or classifier.knn is None:
        raise HTTPException(
            status_code=503, 
            detail="Modèle non chargé"
        )
    
    if not os.path.exists(request.test_directory):
        raise HTTPException(
            status_code=404, 
            detail=f"Dossier de test introuvable: {request.test_directory}"
        )
    
    try:
        # Créer le testeur et lancer l'évaluation
        tester = ModelTester(classifier)
        tester.test_directory(request.test_directory, save_report=True)
        
        # Calcul des métriques globales
        n_total = len(tester.results)
        n_correct = sum(r['correct'] for r in tester.results)
        accuracy = n_correct / n_total * 100 if n_total > 0 else 0
        
        # Statistiques par classe
        class_stats = {}
        for cls in classifier.class_names:
            cls_results = [r for r in tester.results if r['true'] == cls]
            if cls_results:
                cls_correct = sum(r['correct'] for r in cls_results)
                class_stats[cls] = {
                    "n_samples": len(cls_results),
                    "n_correct": cls_correct,
                    "accuracy": cls_correct / len(cls_results) * 100
                }
        
        return {
            "status": "success",
            "accuracy": round(accuracy, 2),
            "n_correct": n_correct,
            "n_total": n_total,
            "class_stats": class_stats,
            "sample_results": tester.results[:5],  # 5 premiers résultats
            "report_saved": True,
            "confusion_matrix_saved": True
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de l'évaluation: {str(e)}"
        )

# =============================================================================
# ROUTES - GESTION DU MODÈLE
# =============================================================================

@app.delete("/model", tags=["Management"])
async def delete_model():
    """
    🗑️ Supprime le modèle sauvegardé
    Utile pour repartir de zéro
    """
    if os.path.exists(Config.MODEL_PATH):
        os.remove(Config.MODEL_PATH)
        return {
            "status": "success",
            "message": "Modèle supprimé avec succès"
        }
    else:
        raise HTTPException(
            status_code=404, 
            detail="Aucun modèle à supprimer"
        )

@app.post("/model/reload", tags=["Management"])
async def reload_model():
    """
    🔄 Recharge le modèle depuis le disque
    Utile après modification manuelle du fichier .pkl
    """
    global classifier
    
    if not os.path.exists(Config.MODEL_PATH):
        raise HTTPException(
            status_code=404, 
            detail="Aucun modèle sauvegardé à recharger"
        )
    
    try:
        classifier.load(Config.MODEL_PATH)
        return {
            "status": "success",
            "message": "Modèle rechargé avec succès",
            "n_references": classifier.stats['n_references'],
            "class_distribution": classifier.stats['class_distribution']
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors du rechargement: {str(e)}"
        )

# =============================================================================
# POINT D'ENTRÉE PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("\n🚀 Démarrage du serveur FastAPI CNN...")
    print("📖 Documentation interactive: http://localhost:8001/docs\n")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",  # Écoute sur toutes les interfaces (0.0.0.0 = accessible de l'extérieur)
        port=8001,  # Port 8001 (différent du Gateway qui est sur 8000)
        reload=True  # Redémarrage auto si le code change (désactiver en production)
    )