# Projet d'Analyse Exploratoire de Données

Ce projet fournit un framework complet pour réaliser des analyses exploratoires de données (EDA) sur des datasets publics ou personnels, en utilisant Python avec pandas, matplotlib et seaborn.

## 🌟 Fonctionnalités

Ce framework d'analyse exploratoire permet de :
- Obtenir un aperçu complet des données (dimensions, types, statistiques)
- Analyser et visualiser les valeurs manquantes
- Réaliser des analyses univariées (distribution des variables)
- Effectuer des analyses bivariées (relations entre variables)
- Calculer et visualiser des matrices de corrélation
- Tester la normalité des distributions
- Analyser des séries temporelles (si applicable)

## 📋 Prérequis

- Python 3.8+
- Bibliothèques requises (voir `requirements.txt`)

## 🚀 Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/votre-nom/data_exploration_project.git
cd data_exploration_project
```

2. Créez un environnement virtuel (recommandé) :
```bash
python -m venv env
source env/bin/activate  # Sur Windows : env\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Installez le package en mode développement :
```bash
pip install -e .
```

## 📊 Utilisation

### En tant que module

```python
from src.analysis.explore_data import analyse_exploratoire
import pandas as pd

# Charger un dataset
df = pd.read_csv("chemin/vers/votre/dataset.csv")

# Lancer l'analyse complète
analyse_exploratoire(df, titre="Analyse de mon dataset")
```

### Via les notebooks

1. Lancez Jupyter Notebook :
```bash
jupyter notebook
```

2. Naviguez vers le dossier `notebooks/` et ouvrez l'un des notebooks.

### Avec des datasets d'exemple

```python
# Exemple avec le dataset Iris
from src.data.load_data import load_iris
from src.analysis.explore_data import analyse_exploratoire

# Charger le dataset Iris
df = load_iris()

# Lancer l'analyse complète
analyse_exploratoire(df, titre="Analyse du dataset Iris")
```

## 📁 Structure du projet

```
data_exploration_project/
├── README.md                       # Documentation principale
├── requirements.txt                # Dépendances Python
├── data/                           # Répertoire des données
│   ├── raw/                        # Données brutes
│   ├── processed/                  # Données nettoyées
│   └── external/                   # Données externes
├── notebooks/                      # Jupyter notebooks
├── src/                            # Code source
│   ├── data/                       # Scripts liés aux données
│   ├── visualization/              # Scripts pour les visualisations
│   └── analysis/                   # Scripts d'analyse
├── reports/                        # Rapports générés
│   └── figures/                    # Figures générées
└── tests/                          # Tests unitaires
```

## 📈 Exemples de visualisations

L'analyse exploratoire génère diverses visualisations, notamment :
- Histogrammes et boxplots pour visualiser les distributions
- Matrices de corrélation avec heatmaps
- Pairplots pour explorer les relations entre variables
- QQ-plots pour tester la normalité
- Graphiques temporels (si applicable)

## 🔍 Datasets recommandés

Voici quelques datasets publics intéressants pour tester ce framework :
- [Iris](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) : Classification de fleurs
- [Titanic](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) : Survie des passagers du Titanic
- [Tips](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv) : Données sur les pourboires au restaurant
- [Boston Housing](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv) : Prix de l'immobilier à Boston

## 👥 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📜 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.