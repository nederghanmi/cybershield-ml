# Guide de Lancement du Projet Machine Learning Cybersécurité
**Auteurs** : Ghanmi Neder & Bel Hadj Ali Nouha  
**Matière** : Machine Learning — 3ème TEK-UP / SSIR  
**Enseignante** : Mme Ahlem Arfaoui  

---

## 1. Description du Projet
Ce projet implémente une solution de détection d'intrusions et d'attaques volumétriques (DDoS) basée sur le jeu de données réel **CICIDS2017** (`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` de 225 745 flux réseau).

Le projet respecte à 100 % le programme officiel du cours :
* **Chapitre 1 : Préparation & EDA** (`data_preprocessing.py`, détection des outliers IQR/Z-score, imputation NaN/Inf, suppression des doublons, mise à l'échelle StandardScaler).
* **Chapitre 2 : Régression & Régularisation** (`train_regression.py`, prédiction de débit, Régression Linéaire, Polynomiale degré 2, Ridge L2, Lasso L1, ElasticNet, avec calcul de R², MAE, RMSE).
* **Chapitre 3 : Classification Supervisée** (`train_classification.py`, comparaison des 5 modèles : Régression Logistique, Arbres de Décision, Random Forest, SVM, k-NN avec validation croisée 5-Fold, matrice de confusion, ROC-AUC).
* **Chapitre 4 : Clustering & PCA** (`train_clustering.py`, K-Means avec méthode du coude WCSS, Dendrogramme CAH Ward, et projection 2D/3D PCA).
* **Détection d'Anomalies** (`anomaly_detection.py`, Isolation Forest non supervisé).
* **Tableau de Bord Interactif (Dashboard Streamlit)** (`app_dashboard.py`).

---

## 2. Commandes d'Exécution

### Étape 1 : Nettoyage et Préparation des données
```bash
python projet_ml_cyber/data_preprocessing.py
```

### Étape 2 : Entraînement de la Classification (5 modèles)
```bash
python projet_ml_cyber/train_classification.py
```

### Étape 3 : Entraînement de la Régression (Régularisation Ridge, Lasso, ElasticNet)
```bash
python projet_ml_cyber/train_regression.py
```

### Étape 4 : Détection d'Anomalies (Isolation Forest & Outliers IQR)
```bash
python projet_ml_cyber/anomaly_detection.py
```

### Étape 5 : Clustering (K-Means, Dendrogramme CAH & PCA)
```bash
python projet_ml_cyber/train_clustering.py
```

### Étape 6 : Lancement du Dashboard Web Interactif
```bash
python -m streamlit run projet_ml_cyber/app_dashboard.py
```
L'application s'ouvre automatiquement sur : **`http://localhost:8501`**
