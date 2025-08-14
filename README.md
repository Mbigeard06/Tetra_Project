# Tetra_Project  
Tetra Project — Détection de Tetrao urogallus sur images panoramiques

Pipeline complet de vision par ordinateur pour détecter des tétras sur des photos panoramiques haute résolution (appareils fixes en milieu forestier). Le projet couvre le prétraitement, la préparation du jeu de données, l’entraînement (YOLO), l’évaluation et l’inférence sur images HD.

✨ Points clés
	•	Tiling SAHI des panoramas en tuiles 640×640, avec mise à jour automatique des boîtes.
	•	Nettoyage des annotations : règles de conservation (aire, ratio), ajout d’images de fond pour réduire les faux positifs.
	•	Entraînement YOLO (v8 / v11) + fine-tuning d’hyperparamètres.
	•	mAP@0.5 ≈ 0.985–0.988 (selon variante) sur dataset corrigé.
	•	Scripts d’automatisation : tri/filtrage des boxes, comparaison visuelle des métriques, visualisation BB.


🔧 Installation
Prérequis : Python 3.10+ et un GPU NVIDIA recommandé (CUDA).
# 1) Cloner
git clone https://github.com/Mbigeard06/Tetra_Project.git
cd Tetra_Project

# 2) Créer l’environnement
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)

# 3) Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt

🗂️ Données (format YOLO)
datasets/
  tetra/
    images/
      train/  val/  test/
    labels/
      train/  val/  test/   # fichiers .txt au format YOLO (cls x_center y_center width height)
      
•	Classes (exemple) : voir classes.txt.
•	Les panoramas bruts sont découpés en tuiles 640×640 avant entraînement.
