 Tetra Project

## 📌 Description
Ce projet a été réalisé dans le cadre d’un stage à la Direction Générale des Systèmes d’Information (DGSI) de l’Institut Agro Dijon.  
L’objectif est de développer un pipeline complet de vision par ordinateur permettant de détecter automatiquement des **Tetrao urogallus** (tétras lyres) sur des images panoramiques haute résolution, capturées par des appareils photo fixes installés dans le Parc naturel régional du Vercors.

Le pipeline prend en charge :
- Le découpage des images panoramiques en tuiles.
- Le nettoyage et la correction des annotations.
- L’entraînement de modèles **YOLO** avec ajustement des hyperparamètres.
- L’inférence et l’évaluation des performances.
- L’automatisation de tâches récurrentes (visualisation, tri des bounding boxes, comparaison de modèles, etc.).

## 🛠️ Technologies utilisées
- **Langage :** Python 3
- **Bibliothèques principales :**
  - [SAHI](https://github.com/obss/sahi) : découpage d’images et gestion des bounding boxes
  - [PyTorch](https://pytorch.org/) : framework deep learning
  - [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) : modèles de détection d’objets
  - Pandas, NumPy, scikit-learn : traitement et analyse de données
  - Matplotlib : visualisation
- **Matériel :**
  - GPU NVIDIA pour l’entraînement
  - Serveur interne de l’Institut Agro Dijon
