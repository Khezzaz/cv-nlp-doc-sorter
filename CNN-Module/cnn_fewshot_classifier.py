import os
import cv2
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# 1. CNN FEATURE EXTRACTOR (RESNET50) - AMÉLIORÉ
# =============================================================================

class CNNFeatureExtractor:
    """
    Extracteur de features basé sur ResNet50 pré-entraîné (ImageNet)
    Sortie : embedding 2048 dimensions
    
    AMÉLIORATIONS:
    - Support de différents backbones (ResNet50, ResNet101, EfficientNet)
    - Extraction par batch pour accélérer
    - Gestion mémoire optimisée
    """

    def __init__(self, 
                 backbone='resnet50',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.backbone = backbone
        
        print(f"🔧 Initialisation du modèle {backbone} sur {device}...")

        # Support de plusieurs backbones
        if backbone == 'resnet50':
            model = models.resnet50(pretrained=True)
            self.feature_dim = 2048
        elif backbone == 'resnet101':
            model = models.resnet101(pretrained=True)
            self.feature_dim = 2048
        elif backbone == 'resnet34':
            model = models.resnet34(pretrained=True)
            self.feature_dim = 512
        else:
            raise ValueError(f"Backbone {backbone} non supporté")

        # Retirer la dernière couche FC
        self.model = nn.Sequential(*list(model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        # Transformations avec data augmentation optionnelle
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extraction pour une seule image"""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(image)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)

        features = features.squeeze().cpu().numpy()
        
        # Normalisation L2 pour stabilité
        features = features / (np.linalg.norm(features) + 1e-8)

        return features

    def extract_features_batch(self, images: List[np.ndarray], 
                               batch_size: int = 8) -> np.ndarray:
        """
        Extraction par batch pour accélérer le traitement
        AMÉLIORATION: Plus rapide pour beaucoup d'images
        """
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            
            # Préparation du batch
            tensors = []
            for img in batch:
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
                tensor = self.transform(pil_img)
                tensors.append(tensor)
            
            batch_tensor = torch.stack(tensors).to(self.device)
            
            # Extraction
            with torch.no_grad():
                features = self.model(batch_tensor)
            
            features = features.squeeze().cpu().numpy()
            
            # Normalisation L2
            if len(features.shape) == 1:
                features = features[np.newaxis, :]
            features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
            
            all_features.append(features)
        
        return np.vstack(all_features)


# =============================================================================
# 2. DOCUMENT PREPROCESSING - AMÉLIORÉ
# =============================================================================

class DocumentPreprocessor:
    """
    AMÉLIORATIONS:
    - Détection automatique de qualité d'image
    - Binarisation adaptative optionnelle
    - Amélioration du contraste
    """

    @staticmethod
    def preprocess_image(image: np.ndarray,
                         target_size=(1024, 1024),
                         enhance_contrast=True,
                         binarize=False) -> np.ndarray:
        """
        Prétraitement avec options avancées
        
        Args:
            enhance_contrast: Améliore le contraste (utile pour scans pâles)
            binarize: Binarisation adaptative (utile pour documents très bruités)
        """

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # AMÉLIORATION: Détection de qualité
        quality_score = DocumentPreprocessor._assess_quality(gray)
        
        # Débruitage adaptatif selon qualité
        if quality_score < 0.5:  # Mauvaise qualité
            gray = cv2.fastNlMeansDenoising(gray, None, 15, 7, 21)
        else:
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # AMÉLIORATION: Amélioration du contraste
        if enhance_contrast:
            gray = DocumentPreprocessor._enhance_contrast(gray)

        # AMÉLIORATION: Binarisation optionnelle
        if binarize:
            gray = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )

        # Deskewing
        gray = DocumentPreprocessor.deskew(gray)
        
        # Reconversion en BGR
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return DocumentPreprocessor.resize_keep_ratio(image, target_size)

    @staticmethod
    def _assess_quality(image: np.ndarray) -> float:
        """
        Évalue la qualité d'une image (0 = mauvaise, 1 = excellente)
        Basé sur la variance du Laplacien (netteté)
        """
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalisation empirique
        quality = min(variance / 500, 1.0)
        return quality

    @staticmethod
    def _enhance_contrast(image: np.ndarray) -> np.ndarray:
        """Amélioration du contraste via CLAHE"""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """Correction d'orientation avec gestion robuste"""
        edges = cv2.Canny(image, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

        if lines is not None and len(lines) > 5:  # Au moins 5 lignes
            angles = []
            for line in lines:
                _, theta = line[0]
                angle = theta * 180 / np.pi - 90
                if -45 < angle < 45:
                    angles.append(angle)

            if len(angles) > 3:  # Suffisamment d'angles valides
                median_angle = np.median(angles)
                
                # Correction uniquement si l'angle est significatif
                if abs(median_angle) > 0.5:
                    h, w = image.shape[:2]
                    M = cv2.getRotationMatrix2D((w // 2, h // 2),
                                                median_angle, 1.0)
                    return cv2.warpAffine(
                        image, M, (w, h),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE
                    )

        return image

    @staticmethod
    def resize_keep_ratio(image: np.ndarray,
                          target_size: Tuple[int, int]) -> np.ndarray:
        """Redimensionnement avec préservation du ratio"""
        h, w = image.shape[:2]
        target_h, target_w = target_size
        ratio = min(target_w / w, target_h / h)

        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h // 2, delta_h - delta_h // 2
        left, right = delta_w // 2, delta_w - delta_w // 2

        return cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )


# =============================================================================
# 3. FEW-SHOT DOCUMENT CLASSIFIER - AMÉLIORÉ
# =============================================================================

class FewShotDocumentClassifier:
    """
    AMÉLIORATIONS:
    - Support de métriques de distance variées
    - Seuil de confiance ajustable
    - Export des résultats en JSON
    - Visualisation des résultats
    """

    def __init__(self, 
                 extractor: CNNFeatureExtractor, 
                 k: int = 3,
                 confidence_threshold: float = 0.4):
        self.extractor = extractor
        self.preprocessor = DocumentPreprocessor()
        self.k = k
        self.confidence_threshold = confidence_threshold

        self.features = []
        self.labels = []
        self.metadata = []

        self.knn = None

        self.class_names = ['CIN', 'Facture', 'Releve_Bancaire']
        
        # Statistiques
        self.stats = {
            'n_references': 0,
            'class_distribution': {},
            'last_updated': None
        }

    def add_references(self, images, labels, metadata, use_batch=True):
        """
        Ajout avec support du traitement par batch
        AMÉLIORATION: Beaucoup plus rapide pour beaucoup d'images
        """
        print(f"📥 Ajout de {len(images)} références...")
        
        # Prétraitement
        preprocessed = [self.preprocessor.preprocess_image(img) for img in images]
        
        # Extraction par batch
        if use_batch and len(images) > 1:
            features = self.extractor.extract_features_batch(preprocessed)
            for feat, lbl, meta in zip(features, labels, metadata):
                self.features.append(feat)
                self.labels.append(lbl)
                self.metadata.append(meta)
        else:
            for img, lbl, meta in zip(preprocessed, labels, metadata):
                feat = self.extractor.extract_features(img)
                self.features.append(feat)
                self.labels.append(lbl)
                self.metadata.append(meta)

        self._build_knn()
        self._update_stats()
        
        print(f"✅ Base: {len(self.features)} exemples")
        print(f"📊 Distribution: {self.stats['class_distribution']}")

    def _build_knn(self):
        """Construction du modèle k-NN"""
        if len(self.features) == 0:
            return
            
        X = np.array(self.features)
        self.knn = NearestNeighbors(
            n_neighbors=min(self.k, len(X)),
            metric='cosine',
            algorithm='brute'
        )
        self.knn.fit(X)

    def _update_stats(self):
        """Mise à jour des statistiques"""
        self.stats['n_references'] = len(self.features)
        self.stats['class_distribution'] = {
            cls: self.labels.count(cls) for cls in self.class_names
        }
        self.stats['last_updated'] = datetime.now().isoformat()

    def classify(self, image: np.ndarray, return_details=True) -> Dict:
        """
        Classification avec détails optionnels
        AMÉLIORATION: Plus d'informations pour debug
        """
        if self.knn is None:
            raise ValueError("Aucune référence chargée!")
        
        # Prétraitement
        img = self.preprocessor.preprocess_image(image)
        
        # Extraction
        feat = self.extractor.extract_features(img)

        # Recherche k-NN
        distances, indices = self.knn.kneighbors([feat])
        sims = 1 - distances[0]

        # Calcul des scores
        scores = {c: 0.0 for c in self.class_names}
        neighbor_info = []

        for idx, sim in zip(indices[0], sims):
            label = self.labels[idx]
            scores[label] += sim
            
            if return_details:
                neighbor_info.append({
                    'label': label,
                    'similarity': float(sim),
                    'distance': float(distances[0][len(neighbor_info)]),
                    'metadata': self.metadata[idx]
                })

        # Normalisation
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        # Prédiction
        pred = max(scores, key=scores.get)
        confidence = scores[pred]
        
        # AMÉLIORATION: Détection de faible confiance
        low_confidence = confidence < self.confidence_threshold

        result = {
            "predicted_class": pred,
            "confidence": confidence,
            "scores": scores,
            "low_confidence_warning": low_confidence
        }
        
        if return_details:
            result["neighbors"] = neighbor_info
            result["avg_similarity"] = float(np.mean(sims))

        return result

    def save(self, path: str):
        """Sauvegarde avec métadonnées"""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        
        save_data = {
            'features': self.features,
            'labels': self.labels,
            'metadata': self.metadata,
            'k': self.k,
            'confidence_threshold': self.confidence_threshold,
            'class_names': self.class_names,
            'stats': self.stats,
            'backbone': self.extractor.backbone
        }
        
        with open(path, "wb") as f:
            pickle.dump(save_data, f)
        
        print(f"💾 Modèle sauvegardé: {path}")

    def load(self, path: str):
        """Chargement avec validation"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Modèle introuvable: {path}")
        
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        self.features = data['features']
        self.labels = data['labels']
        self.metadata = data['metadata']
        self.k = data['k']
        self.confidence_threshold = data.get('confidence_threshold', 0.4)
        self.class_names = data['class_names']
        self.stats = data.get('stats', {})
        
        self._build_knn()
        print(f"📂 Modèle chargé: {len(self.features)} références")


# =============================================================================
# 4. LOAD REFERENCE IMAGES - AMÉLIORÉ
# =============================================================================

def load_reference_images(root="references", verbose=True):
    """
    AMÉLIORATIONS:
    - Support de plus de formats d'images
    - Validation des images
    - Rapport de chargement détaillé
    """
    mapping = {
        "CIN": "CIN",
        "RB": "Releve_Bancaire",
        "FACTURE": "Facture"
    }

    images, labels, metadata = [], [], []
    stats = {cls: 0 for cls in mapping.values()}
    errors = []

    supported_formats = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.jfif')

    for folder, label in mapping.items():
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            if verbose:
                print(f"⚠️  Dossier manquant: {folder_path}")
            continue

        for f in os.listdir(folder_path):
            if f.lower().endswith(supported_formats):
                path = os.path.join(folder_path, f)
                try:
                    img = cv2.imread(path)
                    if img is None:
                        errors.append(f"Erreur lecture: {path}")
                        continue
                    
                    # Validation taille minimale
                    h, w = img.shape[:2]
                    if h < 100 or w < 100:
                        errors.append(f"Image trop petite: {path}")
                        continue
                    
                    images.append(img)
                    labels.append(label)
                    metadata.append({"path": path, "filename": f})
                    stats[label] += 1
                    
                except Exception as e:
                    errors.append(f"Erreur {path}: {str(e)}")

    if verbose:
        print("\n📊 RAPPORT DE CHARGEMENT")
        print(f"✅ Total: {len(images)} images chargées")
        for label, count in stats.items():
            print(f"   - {label}: {count} images")
        
        if errors:
            print(f"\n⚠️  {len(errors)} erreurs:")
            for err in errors[:5]:  # Afficher max 5 erreurs
                print(f"   {err}")

    return images, labels, metadata


# =============================================================================
# 5. TESTING & EVALUATION - NOUVEAU
# =============================================================================

class ModelTester:
    """
    NOUVEAU: Module de test et évaluation
    """
    
    def __init__(self, classifier: FewShotDocumentClassifier):
        self.classifier = classifier
        self.results = []
    
    def test_single_image(self, image_path: str, true_label: Optional[str] = None):
        """Test sur une seule image"""
        if not os.path.exists(image_path):
            print(f"❌ Image introuvable: {image_path}")
            return None
        
        print(f"\n🔍 Test: {os.path.basename(image_path)}")
        
        img = cv2.imread(image_path)
        result = self.classifier.classify(img)
        
        print(f"📄 Prédiction: {result['predicted_class']}")
        print(f"🎯 Confiance: {result['confidence']*100:.2f}%")
        
        if result['low_confidence_warning']:
            print("⚠️  ATTENTION: Confiance faible!")
        
        print(f"\n📊 Scores détaillés:")
        for cls, score in sorted(result['scores'].items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(score * 20)
            print(f"   {cls:20s} {bar} {score*100:.1f}%")
        
        if true_label:
            correct = result['predicted_class'] == true_label
            print(f"\n{'✅ CORRECT' if correct else '❌ INCORRECT'}")
            print(f"   Attendu: {true_label}")
            
            self.results.append({
                'image': image_path,
                'true': true_label,
                'predicted': result['predicted_class'],
                'confidence': result['confidence'],
                'correct': correct
            })
        
        return result
    
    def test_directory(self, test_dir: str, save_report=True):
        """
        Test sur un dossier complet
        Structure attendue: test_dir/CLASS_NAME/*.jpg
        """
        print(f"\n🧪 TEST SUR DOSSIER: {test_dir}")
        
        self.results = []
        
        for class_name in self.classifier.class_names:
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    path = os.path.join(class_dir, f)
                    self.test_single_image(path, true_label=class_name)
        
        # Génération du rapport
        if self.results:
            self._generate_report(save_report)
    
    def _generate_report(self, save=True):
        """Génère un rapport de test complet"""
        print("\n" + "="*60)
        print("📊 RAPPORT DE TEST")
        print("="*60)
        
        n_total = len(self.results)
        n_correct = sum(r['correct'] for r in self.results)
        accuracy = n_correct / n_total * 100
        
        print(f"\n✅ Précision globale: {accuracy:.2f}% ({n_correct}/{n_total})")
        
        # Rapport par classe
        true_labels = [r['true'] for r in self.results]
        pred_labels = [r['predicted'] for r in self.results]
        
        print("\n📋 Rapport détaillé:")
        print(classification_report(true_labels, pred_labels, 
                                   target_names=self.classifier.class_names))
        
        # Matrice de confusion
        self._plot_confusion_matrix(true_labels, pred_labels)
        
        # Sauvegarde JSON
        if save:
            report_path = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump({
                    'accuracy': accuracy,
                    'n_correct': n_correct,
                    'n_total': n_total,
                    'results': self.results,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
            print(f"\n💾 Rapport sauvegardé: {report_path}")
    
    def _plot_confusion_matrix(self, y_true, y_pred):
        """Affiche la matrice de confusion"""
        try:
            cm = confusion_matrix(y_true, y_pred, labels=self.classifier.class_names)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.classifier.class_names,
                       yticklabels=self.classifier.class_names)
            plt.title('Matrice de Confusion')
            plt.ylabel('Vraie Classe')
            plt.xlabel('Classe Prédite')
            
            filename = f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"📈 Matrice de confusion: {filename}")
            plt.close()
        except Exception as e:
            print(f"⚠️  Impossible de générer la matrice: {e}")


# =============================================================================
# 6. MAIN - AMÉLIORÉ
# =============================================================================

def main():
    """
    Fonction principale avec mode interactif
    """
    print("="*60)
    print("🚀 CLASSIFICATEUR CNN - MODE FEW-SHOT")
    print("="*60)
    
    # 1. Initialisation
    extractor = CNNFeatureExtractor(backbone='resnet50')
    classifier = FewShotDocumentClassifier(extractor, k=3, confidence_threshold=0.4)

    # 2. Chargement ou création de la base
    model_path = "models/cnn_fewshot.pkl"
    
    if os.path.exists(model_path):
        print(f"\n📂 Chargement du modèle existant...")
        classifier.load(model_path)
    else:
        print("\n🔹 Création d'une nouvelle base vectorielle...")
        imgs, lbls, meta = load_reference_images("references")
        
        if len(imgs) == 0:
            print("❌ Aucune image de référence trouvée!")
            print("📁 Structure attendue:")
            print("   references/")
            print("   ├── CIN/")
            print("   ├── FACTURE/")
            print("   └── RB/")
            return
        
        classifier.add_references(imgs, lbls, meta)
        classifier.save(model_path)

    # 3. Menu interactif
    while True:
        print("\n" + "="*60)
        print("MENU")
        print("="*60)
        print("1. Tester une image")
        print("2. Tester un dossier")
        print("3. Afficher les statistiques")
        print("4. Quitter")
        
        choice = input("\nChoix: ").strip()
        
        if choice == '1':
            path = input("Chemin de l'image: ").strip()
            tester = ModelTester(classifier)
            tester.test_single_image(path)
        
        elif choice == '2':
            test_dir = input("Chemin du dossier de test: ").strip()
            tester = ModelTester(classifier)
            tester.test_directory(test_dir)
        
        elif choice == '3':
            print("\n📊 STATISTIQUES")
            print(f"Références totales: {classifier.stats['n_references']}")
            print(f"Distribution:")
            for cls, count in classifier.stats['class_distribution'].items():
                print(f"   - {cls}: {count}")
            print(f"Dernière MAJ: {classifier.stats.get('last_updated', 'N/A')}")
        
        elif choice == '4':
            print("👋 Au revoir!")
            break


if __name__ == "__main__":
    main()