# 🩺 Prédiction du Diabète par Machine Learning

![1](https://github.com/user-attachments/assets/b019fa20-0efc-44e3-9c32-145c83ddf001)
![2](https://github.com/user-attachments/assets/8dfaaf8e-118a-4f43-a74f-ce15ae1c4a6c)

## 🎯 Aperçu du Projet

Ce projet vise à développer des modèles de machine learning pour la prédiction précoce du diabète de type 2. En utilisant des techniques avancées de data mining et d'apprentissage automatique, nous identifions les facteurs de risque principaux et construisons des classificateurs performants pour le dépistage.

**Performance du meilleur modèle : F1-Score = 0.8845**

## 🎯 Objectifs

- **Identifier** les facteurs de risque cliniques associés au diabète
- **Développer** des modèles prédictifs performants utilisant plusieurs algorithmes
- **Évaluer** comparativement les performances des différents modèles
- **Fournir** des insights actionnables pour la prévention du diabète

## 📊 Dataset

### Caractéristiques
- **100,000 observations** initiales
- **9 variables** cliniques et démographiques
- **Taux de prévalence** : 8.5% (cohérent avec les données mondiales)

### Variables Utilisées

| Variable | Type | Description |
|----------|------|-------------|
| `gender` | Catégorielle | Genre du patient |
| `age` | Continue | Âge en années |
| `hypertension` | Binaire | Présence d'hypertension |
| `heart_disease` | Binaire | Antécédents cardiaques |
| `smoking_history` | Catégorielle | Historique tabagique |
| `bmi` | Continue | Indice de masse corporelle |
| `HbA1c_level` | Continue | Hémoglobine glyquée (%) |
| `blood_glucose_level` | Continue | Glucose sanguin (mg/dL) |
| `diabetes` | Binaire | Variable cible |

## 🔬 Méthodologie

### Approche CRISP-DM
Le projet suit rigoureusement la méthodologie CRISP-DM :

1. **Compréhension des Données** - Exploration et analyse descriptive
2. **Préparation** - Nettoyage, feature engineering, normalisation
3. **Modélisation** - Entraînement de 5 algorithmes différents
4. **Évaluation** - Validation comparative des performances
5. **Déploiement** - Sauvegarde du meilleur modèle

### Gestion du Déséquilibre
- **Technique** : SMOTE (Synthetic Minority Over-sampling Technique)
- **Ratio initial** : 10:1 (non-diabétiques:diabétiques)
- **Stratégie** : Rééchantillonnage pour équilibrer les classes

## ⚙️ Installation
```bash
# Cloner le repository
git clone https://github.com/yaelaya/diabetes-prediction-ml.git
```


