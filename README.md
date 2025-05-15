# Projet d'Analyse Exploratoire de Données Musicales

[![Website](https://img.shields.io/badge/Website-View%20Demo-blue)](https://votre-nom.github.io/eda-music/)

Ce projet fournit un framework complet pour réaliser des analyses exploratoires de données (EDA) sur des datasets musicaux, notamment Spotify, en utilisant Python avec pandas, matplotlib et seaborn.

🌐 **[Voir la démonstration en ligne](https://votre-nom.github.io/eda-music/)**

## 🌟 Fonctionnalités

Ce framework d'analyse exploratoire permet de :
- Obtenir un aperçu complet des données (dimensions, types, statistiques)
- Analyser et visualiser les valeurs manquantes
- Réaliser des analyses univariées (distribution des variables)
- Effectuer des analyses bivariées (relations entre variables)
- Calculer et visualiser des matrices de corrélation
- Tester la normalité des distributions
- Analyser des séries temporelles (si applicable)
- **Spécialisation Spotify** : Analysez en profondeur les données musicales (caractéristiques audio, popularité, genres)

### Analyses Spotify spécialisées
- Corrélations entre caractéristiques audio (danceability, energy, acousticness, etc.)
- Distributions des métriques audio par genre musical
- Facteurs influençant la popularité des artistes et des chansons
- Tendances temporelles des caractéristiques musicales
- Exploration des différences entre genres musicaux
- Analyse des artistes les plus populaires

## 📋 Prérequis

- Python 3.8+
- Bibliothèques requises (voir `requirements.txt`)

## 🚀 Installation

1. Clonez ce dépôt :
```bash
git clone https://github.com/votre-nom/eda-music.git
cd eda-music
```

2. Créez un environnement virtuel (recommandé) :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate
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

### Via le script d'analyse Spotify

Le moyen le plus simple d'analyser le dataset Spotify est d'utiliser le script dédié :

```bash
# Exécute l'analyse complète et sauvegarde les figures
./run_spotify_analysis.sh

# Alternativement, vous pouvez exécuter directement le script Python
python spotify_analysis.py --save-figures
```

### En tant que module Python

```python
from src.data.load_data import load_spotify
from src.analysis.spotify_analysis import analyser_spotify_dataset

# Charger le dataset Spotify
spotify_df = load_spotify()

# Lancer l'analyse complète des données Spotify
analyser_spotify_dataset(spotify_df)
```

### Analyses personnalisées

```python
from src.data.load_data import load_spotify
from src.analysis.spotify_analysis import (
    analyser_correlations_audio,
    analyser_par_genre,
    analyser_popularite,
    analyser_tendances_temporelles
)

# Charger le dataset
df = load_spotify()

# Analyser les corrélations entre caractéristiques audio
figures_corr = analyser_correlations_audio(df)

# Analyser les différences entre genres musicaux
figures_genre, moyennes_par_genre = analyser_par_genre(df, n_genres=10)

# Analyser les facteurs de popularité
figures_pop = analyser_popularite(df)

# Analyser l'évolution temporelle des caractéristiques musicales
figures_temps = analyser_tendances_temporelles(df)
```

### Via les notebooks Jupyter

1. Lancez Jupyter Notebook :
```bash
jupyter notebook
```

2. Naviguez vers le dossier `notebooks/` et ouvrez `spotify_eda.ipynb` pour une analyse interactive complète.

## 📁 Structure du projet

```
eda-music/
├── README.md                       # Documentation principale
├── requirements.txt                # Dépendances Python
├── setup.py                        # Configuration du package
├── spotifydataset.csv              # Données Spotify
├── spotify_analysis.py             # Script d'analyse principal
├── run_spotify_analysis.sh         # Script shell d'automatisation
├── data/                           # Répertoire des données
│   ├── raw/                        # Données brutes
│   ├── processed/                  # Données nettoyées
├── notebooks/                      # Jupyter notebooks
│   └── spotify_eda.ipynb           # Notebook pour l'analyse Spotify
├── src/                            # Code source
│   ├── data/                       # Scripts liés aux données
│   │   └── load_data.py            # Fonctions de chargement
│   ├── visualization/              # Scripts pour les visualisations
│   │   └── visualize.py            # Fonctions de visualisation
│   ├── analysis/                   # Scripts d'analyse
│   │   ├── explore_data.py         # Module d'analyse générique
│   │   └── spotify_analysis.py     # Module d'analyse Spotify
├── reports/                        # Rapports générés
│   └── figures/                    # Figures générées
├── docs/                           # Documentation détaillée
│   └── spotify_dataset.md          # Description du dataset Spotify
└── tests/                          # Tests unitaires
```

## 📊 Métriques audio analysées

| Métrique | Description |
|----------|-------------|
| Danceability | Aptitude du morceau à la danse (0.0 à 1.0) |
| Energy | Intensité et activité perçues (0.0 à 1.0) |
| Acousticness | Probabilité que le morceau soit acoustique (0.0 à 1.0) |
| Valence | Positivité musicale du morceau (0.0 à 1.0) |
| Instrumentalness | Probabilité d'absence de voix (0.0 à 1.0) |
| Liveness | Détecte la présence d'un public (0.0 à 1.0) |
| Speechiness | Présence de mots parlés (0.0 à 1.0) |
| Tempo | Rythme estimé en BPM |
| Loudness | Volume global en dB (-60 à 0) |
| Key | Tonalité du morceau (0=C, 1=C#, etc.) |
| Mode | Modalité (0=mineur, 1=majeur) |

## 🔍 Autres datasets compatibles

Le projet permet également de charger automatiquement:
- [Iris](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv) : Classification de fleurs
- [Titanic](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) : Survie des passagers du Titanic
- [Tips](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv) : Données sur les pourboires au restaurant
- [Boston Housing](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv) : Prix de l'immobilier à Boston

## 👥 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## 📜 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.