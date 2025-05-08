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
- **NOUVEAU** : Analyser les données musicales de Spotify (caractéristiques audio, popularité, genres)

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

### Pour l'analyse du dataset Spotify

```python
from src.data.load_data import load_spotify
from src.analysis.spotify_analysis import analyser_spotify_dataset

# Charger le dataset Spotify
spotify_df = load_spotify()

# Lancer l'analyse complète des données Spotify
analyser_spotify_dataset(spotify_df)
```

### Via les notebooks

1. Lancez Jupyter Notebook :
```bash
jupyter notebook
```

2. Naviguez vers le dossier `notebooks/` et ouvrez l'un des notebooks:
   - `spotify_eda.ipynb` pour l'analyse des données Spotify
   - Autres notebooks disponibles pour d'autres types d'analyse

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
├── spotifydataset.csv              # Données Spotify
├── data/                           # Répertoire des données
│   ├── raw/                        # Données brutes
│   ├── processed/                  # Données nettoyées
│   └── external/                   # Données externes
├── notebooks/                      # Jupyter notebooks
│   └── spotify_eda.ipynb           # Notebook pour l'analyse Spotify
├── src/                            # Code source
│   ├── data/                       # Scripts liés aux données
│   ├── visualization/              # Scripts pour les visualisations
│   └── analysis/                   # Scripts d'analyse
│       └── spotify_analysis.py     # Module d'analyse spécifique à Spotify
├── reports/                        # Rapports générés
│   └── figures/                    # Figures générées
└── tests/                          # Tests unitaires
```

## 📈 Analyse du dataset Spotify

Notre analyse du dataset Spotify comprend :

- **Analyse des artistes** : Popularité, nombre de followers, genres associés
- **Analyse des métriques audio** : Dansabilité, énergie, acoustique, etc.
- **Analyse par genre musical** : Caractéristiques spécifiques à chaque genre
- **Analyse de popularité** : Facteurs influençant la popularité des artistes et des chansons
- **Analyse temporelle** : Évolution des caractéristiques musicales au fil du temps

### Métriques audio analysées

| Métrique | Description |
|----------|-------------|
| Danceability | Mesure l'aptitude du morceau à la danse (0.0 à 1.0) |
| Energy | Mesure de l'intensité et de l'activité (0.0 à 1.0) |
| Acousticness | Probabilité que le morceau soit acoustique (0.0 à 1.0) |
| Valence | Positivité musicale du morceau (0.0 à 1.0) |
| Instrumentalness | Probabilité d'absence de voix (0.0 à 1.0) |
| Liveness | Détecte la présence d'un public (0.0 à 1.0) |
| Speechiness | Présence de mots parlés (0.0 à 1.0) |
| Tempo | Rythme estimé en BPM |

## 🔍 Datasets inclus et recommandés

Le projet inclut:
- **Spotify** : Données sur les artistes, chansons et caractéristiques audio (`spotifydataset.csv`)

Et permet de charger automatiquement:
- [Iris](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) : Classification de fleurs
- [Titanic](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) : Survie des passagers du Titanic
- [Tips](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv) : Données sur les pourboires au restaurant
- [Boston Housing](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv) : Prix de l'immobilier à Boston

## 👥 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📜 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.