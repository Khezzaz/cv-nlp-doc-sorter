"""
Système OCR FastAPI pour Classification de Documents Administratifs
Module NLP - Version Finale avec Support Dictionnaire
Équipe KHEZZAZ, ED-DALHI, HAYTOM
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import cv2
import numpy as np
from paddleocr import PaddleOCR
import re
from pathlib import Path
import json
from datetime import datetime
import logging
import tempfile

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# INITIALISATION PADDLEOCR
# =============================================================================

ocr_engine = None

def initialize_ocr():
    """Initialise PaddleOCR"""
    global ocr_engine
    if ocr_engine is None:
        logger.info("Initialisation de PaddleOCR...")
        ocr_engine = PaddleOCR(lang='fr')
        logger.info("PaddleOCR initialisé avec succès")
    return ocr_engine

# =============================================================================
# GESTION DU CYCLE DE VIE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_ocr()
    logger.info("API OCR prête")
    yield
    logger.info("API OCR arrêtée")

# =============================================================================
# APPLICATION FASTAPI
# =============================================================================

app = FastAPI(
    title="API OCR - Classification Documents",
    description="Module NLP pour extraction et classification de texte",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DICTIONNAIRES DE MOTS-CLÉS
# =============================================================================

KEYWORDS_DATABASE = {
    "CIN": {
        "mots_obligatoires": [
            "royaume", "maroc", "identité", "nationale", "carte", 
            "nationalité", "marocaine", "nom", "prénom", "sexe"
        ],
        "regex_patterns": [
            r"[A-Z]{1,2}\d{6,7}",
            r"\d{2}[/-]\d{2}[/-]\d{4}",
        ],
        "poids": 3.0
    },
    "FACTURE": {
        "mots_obligatoires": [
            "facture", "tva", "montant", "total", "ht", "ttc",
            "consommation", "kwh", "abonnement", "période"
        ],
        "mots_specifiques": {
            "electricite": ["lydec", "redal", "amendis", "électricité", "kwh", "puissance"],
            "eau": ["eau", "assainissement", "m3", "index"],
            "telecom": ["iam", "orange", "inwi", "appels", "internet", "forfait"]
        },
        "regex_patterns": [
            r"ICE\s*:\s*\d{15}",
            r"RC\s*:\s*\d+",
            r"\d+[,\.]\d{2}\s*(DH|MAD)",
        ],
        "poids": 2.5
    },
    "RELEVE_BANCAIRE": {
        "mots_obligatoires": [
            "relevé", "compte", "solde", "opération", "crédit", 
            "débit", "virement", "banque", "rib"
        ],
        "mots_specifiques": {
            "banques": ["attijariwafa", "bmce", "bmci", "crédit agricole", "cih", "sgmb"]
        },
        "regex_patterns": [
            r"\d{16,24}",
            r"[+-]?\d+[,\.]\d{2}",
        ],
        "poids": 2.8
    }
}

# =============================================================================
# MODÈLES DE DONNÉES
# =============================================================================

class OCRResult(BaseModel):
    texte_brut: str
    texte_nettoye: str
    nombre_mots: int
    confiance_moyenne: float
    
class ClassificationResult(BaseModel):
    categorie: str
    confiance: float
    scores_details: Dict[str, float]
    mots_cles_detectes: Dict[str, List[str]]
    
class DocumentAnalysis(BaseModel):
    ocr_result: OCRResult
    classification: ClassificationResult
    metadata: Dict
    timestamp: str

# =============================================================================
# PRÉTRAITEMENT D'IMAGES
# =============================================================================

class ImagePreprocessor:
    @staticmethod
    def preprocess_image(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        return binary

# =============================================================================
# EXTRACTION TEXTE OCR - GESTION FORMAT DICTIONNAIRE
# =============================================================================

def extract_text_from_paddleocr_result(result) -> tuple:
    """
    Extrait le texte des résultats PaddleOCR
    Format: result[0] contient les clés 'rec_texts' et 'rec_scores'
    """
    texte_brut = ""
    confiances = []
    
    try:
        if not result or len(result) == 0:
            logger.warning("Résultat OCR vide")
            return "", []
        
        # Récupérer les données
        data = result[0]
        
        # CAS 1: Format DICTIONNAIRE avec 'rec_texts' et 'rec_scores'
        if isinstance(data, dict):
            # Chercher les textes dans 'rec_texts'
            if 'rec_texts' in data and 'rec_scores' in data:
                texts = data['rec_texts']
                scores = data['rec_scores']
                
                logger.info(f"Trouvé {len(texts)} textes")
                
                for text, score in zip(texts, scores):
                    texte_brut += str(text) + " "
                    confiances.append(float(score))
            
            # Fallback: chercher d'autres clés possibles
            else:
                logger.warning(f"Clés 'rec_texts' non trouvées. Clés disponibles: {list(data.keys())}")
                
                for key in ['rec_text', 'text', 'texts', 'lines', 'result']:
                    if key in data:
                        text_data = data[key]
                        if isinstance(text_data, list):
                            for item in text_data:
                                texte_brut += str(item) + " "
                                confiances.append(0.9)
                        break
        
        # CAS 2: Format LISTE (ancienne version)
        elif isinstance(data, list):
            logger.info("Format liste détecté")
            for line in data:
                try:
                    if line and len(line) >= 2:
                        text = str(line[1][0])
                        confidence = float(line[1][1])
                        texte_brut += text + " "
                        confiances.append(confidence)
                except:
                    continue
        
        else:
            logger.error(f"Format inconnu: {type(data)}")
    
    except Exception as e:
        logger.error(f"Erreur extraction: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    return texte_brut, confiances

def perform_ocr_on_image(image: np.ndarray) -> tuple:
    """Effectue l'OCR sur une image"""
    try:
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image)
        
        # OCR
        ocr = initialize_ocr()
        result = ocr.ocr(tmp_path)
        
        # Supprimer le fichier temporaire
        try:
            Path(tmp_path).unlink()
        except:
            pass
        
        # Extraire le texte
        texte, confiances = extract_text_from_paddleocr_result(result)
        
        return texte, confiances
        
    except Exception as e:
        logger.error(f"Erreur OCR: {e}")
        return "", []

# =============================================================================
# CLASSIFICATEUR
# =============================================================================

class TextClassifier:
    def __init__(self):
        self.keywords_db = KEYWORDS_DATABASE
    
    def normalize_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_keywords(self, text: str, category: str) -> List[str]:
        found = []
        normalized = self.normalize_text(text)
        
        data = self.keywords_db.get(category, {})
        for keyword in data.get("mots_obligatoires", []):
            if keyword.lower() in normalized:
                found.append(keyword)
        
        for subcat, words in data.get("mots_specifiques", {}).items():
            for word in words:
                if word.lower() in normalized:
                    found.append(f"{word} ({subcat})")
        
        return found
    
    def calculate_keyword_score(self, text: str, category: str) -> float:
        normalized = self.normalize_text(text)
        data = self.keywords_db.get(category, {})
        mots = data.get("mots_obligatoires", [])
        
        score = sum(1 for k in mots if k.lower() in normalized)
        return score / len(mots) if mots else 0
    
    def calculate_pattern_score(self, text: str, category: str) -> float:
        data = self.keywords_db.get(category, {})
        patterns = data.get("regex_patterns", [])
        
        if not patterns:
            return 0.0
        
        matches = sum(1 for p in patterns if re.search(p, text, re.IGNORECASE))
        return matches / len(patterns)
    
    def classify(self, text: str) -> Dict:
        scores = {}
        keywords_found = {}
        
        for category in self.keywords_db.keys():
            kw_score = self.calculate_keyword_score(text, category)
            pat_score = self.calculate_pattern_score(text, category)
            weight = self.keywords_db[category].get("poids", 1.0)
            
            final_score = (kw_score * 0.7 + pat_score * 0.3) * weight
            scores[category] = final_score
            keywords_found[category] = self.extract_keywords(text, category)
        
        best_category = max(scores, key=scores.get) if scores else "INCONNU"
        confidence = scores.get(best_category, 0.0)
        
        return {
            "categorie": best_category,
            "confiance": confidence,
            "scores_details": scores,
            "mots_cles_detectes": keywords_found
        }

# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "API OCR - Classification de Documents",
        "version": "1.0.0",
        "ocr_engine": "PaddleOCR",
        "endpoints": {
            "health": "/health",
            "extract_text": "/ocr/extract",
            "extract_with_boxes": "/ocr/extract_with_boxes",
            "classify_document": "/ocr/classify",
            "full_analysis": "/ocr/analyze"
        }
    }

@app.get("/health")
async def health_check():
    ocr = initialize_ocr()
    return {
        "status": "healthy",
        "ocr_engine": "PaddleOCR" if ocr else "Non initialisé",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/ocr/extract_with_boxes")
async def extract_text_with_boxes(file: UploadFile = File(...)):
    """
    Extraction du texte avec les coordonnées des bounding boxes
    
    Args:
        file: Image du document
    
    Returns:
        Liste de blocs de texte avec leurs coordonnées (xmin, ymin, xmax, ymax)
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Image invalide")
        
        # Sauvegarder temporairement
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp_path = tmp.name
            cv2.imwrite(tmp_path, image)
        
        # OCR
        ocr = initialize_ocr()
        result = ocr.ocr(tmp_path)
        
        # Supprimer le fichier temporaire
        try:
            Path(tmp_path).unlink()
        except:
            pass
        
        # Extraire les blocs de texte avec leurs coordonnées
        text_blocks = []
        
        if result and len(result) > 0:
            data = result[0]
            
            # Si c'est un dictionnaire (nouvelle version PaddleOCR)
            if isinstance(data, dict):
                texts = data.get('rec_texts', [])
                scores = data.get('rec_scores', [])
                polys = data.get('rec_polys', [])
                
                for idx, (text, score, poly) in enumerate(zip(texts, scores, polys)):
                    # Extraire les coordonnées min et max
                    poly_array = np.array(poly)
                    xmin = int(poly_array[:, 0].min())
                    xmax = int(poly_array[:, 0].max())
                    ymin = int(poly_array[:, 1].min())
                    ymax = int(poly_array[:, 1].max())
                    
                    text_blocks.append({
                        "id": idx,
                        "text": str(text),
                        "confidence": float(score),
                        "bbox": {
                            "xmin": xmin,
                            "ymin": ymin,
                            "xmax": xmax,
                            "ymax": ymax
                        },
                        "polygon": poly_array.tolist()  # Coordonnées complètes du polygone
                    })
            
            # Si c'est une liste (ancienne version)
            elif isinstance(data, list):
                for idx, line in enumerate(data):
                    try:
                        if line and len(line) >= 2:
                            bbox = line[0]  # Coordonnées du rectangle
                            text_info = line[1]
                            
                            text = str(text_info[0])
                            confidence = float(text_info[1])
                            
                            # Extraire xmin, ymin, xmax, ymax
                            bbox_array = np.array(bbox)
                            xmin = int(bbox_array[:, 0].min())
                            xmax = int(bbox_array[:, 0].max())
                            ymin = int(bbox_array[:, 1].min())
                            ymax = int(bbox_array[:, 1].max())
                            
                            text_blocks.append({
                                "id": idx,
                                "text": text,
                                "confidence": confidence,
                                "bbox": {
                                    "xmin": xmin,
                                    "ymin": ymin,
                                    "xmax": xmax,
                                    "ymax": ymax
                                },
                                "polygon": bbox_array.tolist()
                            })
                    except Exception as e:
                        logger.warning(f"Erreur ligne {idx}: {e}")
                        continue
        
        return {
            "total_blocks": len(text_blocks),
            "image_shape": {
                "height": image.shape[0],
                "width": image.shape[1]
            },
            "text_blocks": text_blocks,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur extraction avec boxes: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/extract", response_model=OCRResult)
async def extract_text(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Image invalide")
        
        preprocessor = ImagePreprocessor()
        processed = preprocessor.preprocess_image(image)
        
        texte_brut, confiances = perform_ocr_on_image(processed)
        
        texte_nettoye = re.sub(r'\s+', ' ', texte_brut).strip()
        nombre_mots = len(texte_nettoye.split()) if texte_nettoye else 0
        confiance_moyenne = float(np.mean(confiances)) if confiances else 0.0
        
        return OCRResult(
            texte_brut=texte_brut,
            texte_nettoye=texte_nettoye,
            nombre_mots=nombre_mots,
            confiance_moyenne=confiance_moyenne
        )
        
    except Exception as e:
        logger.error(f"Erreur extraction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/classify", response_model=ClassificationResult)
async def classify_text(file: UploadFile = File(...)):
    try:
        ocr_result = await extract_text(file)
        classifier = TextClassifier()
        classification = classifier.classify(ocr_result.texte_nettoye)
        return ClassificationResult(**classification)
    except Exception as e:
        logger.error(f"Erreur classification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/analyze", response_model=DocumentAnalysis)
async def full_analysis(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Image invalide")
        
        preprocessor = ImagePreprocessor()
        processed = preprocessor.preprocess_image(image)
        
        texte_brut, confiances = perform_ocr_on_image(processed)
        
        texte_nettoye = re.sub(r'\s+', ' ', texte_brut).strip()
        nombre_mots = len(texte_nettoye.split()) if texte_nettoye else 0
        confiance_moyenne = float(np.mean(confiances)) if confiances else 0.0
        
        ocr_result = OCRResult(
            texte_brut=texte_brut,
            texte_nettoye=texte_nettoye,
            nombre_mots=nombre_mots,
            confiance_moyenne=confiance_moyenne
        )
        
        classifier = TextClassifier()
        classification = classifier.classify(texte_nettoye)
        classification_result = ClassificationResult(**classification)
        
        metadata = {
            "filename": file.filename,
            "image_shape": list(image.shape),
            "preprocessing_applied": True
        }
        
        return DocumentAnalysis(
            ocr_result=ocr_result,
            classification=classification_result,
            metadata=metadata,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Erreur analyse: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)