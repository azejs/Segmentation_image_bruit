# Segmentation_image_bruit

Projet de segmentation d'images en présence de bruit, combinant des techniques classiques de traitement d'image et des approches d'apprentissage profond.

## 📋 Description

Ce projet propose des méthodes de segmentation d'images robustes au bruit, en s'appuyant sur des bibliothèques de traitement d'image (OpenCV, scikit-image) et de deep learning (PyTorch).

## 🚀 Installation

### Prérequis

- Python 3.9 ou supérieur
- pip

### Installation des dépendances

Clonez le dépôt puis installez les dépendances via le fichier `requirements.txt` :

```bash
git clone <url_du_depot>
cd Segmentation_image_bruit
pip install -r requirements.txt
```

Il est recommandé d'utiliser un environnement virtuel :

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

## 📦 Dépendances principales

| Bibliothèque | Version minimale | Usage |
|---|---|---|
| numpy | 1.24.0 | Calcul numérique |
| opencv-python | 4.8.0 | Traitement d'image |
| scikit-image | 0.21.0 | Algorithmes de segmentation |
| scikit-learn | 1.3.0 | Machine learning classique |
| matplotlib | 3.7.0 | Visualisation |
| Pillow | 10.0.0 | Manipulation d'images |
| torch | 2.0.0 | Deep learning |
| torchvision | 0.15.0 | Modèles et datasets vision |
| tqdm | 4.65.0 | Barres de progression |
| pandas | 2.0.0 | Manipulation de données |
| seaborn | 0.12.0 | Visualisation statistique |
| jupyter | 1.0.0 | Notebooks interactifs |
| ipykernel | 6.0.0 | Noyau Jupyter |
| scipy | 1.11.0 | Calcul scientifique |

## 📁 Structure du projet

```
Segmentation_image_bruit/
├── data/               # Jeux de données (images bruitées, masques, etc.)
├── notebooks/          # Notebooks Jupyter d'exploration et d'expérimentation
├── src/                # Code source (modèles, prétraitement, utils)
├── results/            # Résultats et visualisations
├── requirements.txt    # Dépendances du projet
└── README.md
```

## ▶️ Utilisation

```bash
jupyter notebook
```

Puis ouvrez le notebook souhaité dans le dossier `notebooks/`.

## 🧪 Tests

Ajoutez ici les instructions pour lancer les tests si le projet en contient.

## 👤 Auteur

Ingénieur en systèmes d'information – École SUPMTI ✅

## 📄 Licence

À compléter selon la licence choisie pour le projet.
