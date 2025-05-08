# Documentation du Dataset Spotify

## Description

Ce dataset contient des informations sur des artistes, des chansons, et leurs caractéristiques audio sur Spotify. Il permet d'analyser les tendances musicales, les caractéristiques des genres et les facteurs qui influencent la popularité.

## Structure et contenu

Le fichier `spotifydataset.csv` contient environ 1000 lignes, chacune représentant une chanson avec ses métadonnées d'artiste et ses caractéristiques audio.

## Colonnes

| Colonne | Type | Description |
|---------|------|-------------|
| `artist_name` | texte | Nom de l'artiste |
| `genres` | texte | Genres associés à l'artiste (format: liste sous forme de chaîne) |
| `followers` | nombre | Nombre d'abonnés de l'artiste sur Spotify |
| `artist_popularity` | nombre | Indice de popularité de l'artiste (0-100) |
| `artist_url` | texte | URL Spotify de l'artiste |
| `track_name` | texte | Nom de la chanson |
| `album_name` | texte | Nom de l'album contenant la chanson |
| `release_date` | date | Date de sortie de l'album |
| `duration_ms` | nombre | Durée de la chanson en millisecondes |
| `explicit` | booléen | Indique si la chanson contient des paroles explicites |
| `track_popularity` | nombre | Indice de popularité de la chanson (0-100) |
| `danceability` | nombre | Aptitude de la chanson à la danse (0.0-1.0) |
| `energy` | nombre | Intensité et activité perçues (0.0-1.0) |
| `key` | nombre | Tonalité de la chanson (0=C, 1=C#, etc.) |
| `loudness` | nombre | Volume global en dB (-60 à 0) |
| `mode` | nombre | Modalité de la chanson (0=mineur, 1=majeur) |
| `speechiness` | nombre | Présence de mots parlés (0.0-1.0) |
| `acousticness` | nombre | Confiance que la chanson soit acoustique (0.0-1.0) |
| `instrumentalness` | nombre | Probabilité d'absence de voix (0.0-1.0) |
| `liveness` | nombre | Présence d'un public/concert (0.0-1.0) |
| `valence` | nombre | Positivité musicale (0.0-1.0) |
| `tempo` | nombre | Rythme estimé en BPM |

## Caractéristiques Audio Expliquées

### Danceability (Aptitude à la danse)
Mesure à quel point une chanson est adaptée à la danse, basée sur des éléments musicaux comme le tempo, la stabilité du rythme, la force de la pulsation et la régularité globale. Valeur de 0.0 (moins dansant) à 1.0 (plus dansant).

### Energy (Énergie)
Mesure représentant l'intensité et l'activité perçues. Les pistes énergiques sont généralement rapides, fortes et bruyantes. Par exemple, le death metal a une forte énergie, tandis qu'un prélude de Bach a un score plus bas. Valeur de 0.0 à 1.0.

### Key (Tonalité)
Tonalité globale du morceau. Les entiers sont associés aux tonalités standard, par exemple 0 = Do, 1 = Do#, 2 = Ré, etc.

### Loudness (Volume)
Volume global d'une piste en décibels (dB). Les valeurs typiques vont de -60 à 0 dB.

### Mode (Mode)
Mode (majeur ou mineur) de la piste. 1 représente le mode majeur, 0 représente le mode mineur.

### Speechiness (Présence de paroles)
Détecte la présence de mots parlés dans une piste. Plus la valeur est proche de 1.0, plus le morceau est verbal (podcast, discours). Au-dessus de 0.66, la piste est probablement entièrement parlée. Entre 0.33 et 0.66, il y a un mélange de musique et de paroles. En dessous de 0.33, il s'agit probablement de musique instrumentale.

### Acousticness (Caractère acoustique)
Mesure de confiance (0.0 à 1.0) qu'une piste soit acoustique. 1.0 représente une haute confiance que la piste est acoustique.

### Instrumentalness (Caractère instrumental)
Prédit si une piste ne contient pas de voix. Plus la valeur est proche de 1.0, plus il est probable que la piste ne contienne pas de contenu vocal. Les "ooh" et "aah" sont traités comme des instruments. Les pistes rap ou parlées seront clairement "vocales".

### Liveness (Présence d'un public)
Détecte la présence d'un public dans l'enregistrement. Une valeur élevée indique une forte probabilité que la piste ait été enregistrée en direct. Les valeurs au-dessus de 0.8 indiquent une forte probabilité que la piste soit live.

### Valence (Valence)
Mesure de 0.0 à 1.0 décrivant la positivité musicale. Les pistes à haute valence sonnent plus positives (joyeuses, enjouées), tandis que les pistes à basse valence sonnent plus négatives (tristes, en colère).

### Tempo (Tempo)
Tempo estimé d'une piste en battements par minute (BPM).

## Utilisation avec notre Framework

### Chargement des données

```python
from src.data.load_data import load_spotify

# Charger le dataset
spotify_df = load_spotify()

# Afficher les informations de base
print(f"Dimensions: {spotify_df.shape}")
print(spotify_df.info())
spotify_df.head()
```

### Analyse complète

```python
from src.analysis.spotify_analysis import analyser_spotify_dataset

# Lancer l'analyse complète
analyser_spotify_dataset(spotify_df)
```

### Analyses spécifiques

```python
from src.analysis.spotify_analysis import (
    analyser_correlations_audio,
    analyser_par_genre,
    analyser_popularite,
    analyser_distribution_audio,
    analyser_tendances_temporelles,
    analyser_top_artistes
)

# Analyse des corrélations entre caractéristiques audio
figures_corr = analyser_correlations_audio(spotify_df)

# Analyse par genre musical
figures_genre, moyennes_par_genre = analyser_par_genre(spotify_df, n_genres=15)

# Analyse de popularité
figures_pop = analyser_popularite(spotify_df)

# Analyse de la distribution des caractéristiques audio
figures_dist = analyser_distribution_audio(spotify_df)

# Analyse des tendances temporelles
figures_temps = analyser_tendances_temporelles(spotify_df)

# Analyse des artistes les plus populaires
figures_artistes = analyser_top_artistes(spotify_df, n_artistes=10)
```

### Visualisations avancées

```python
from src.visualization.visualize import (
    spotify_feature_distribution,
    spotify_correlation_heatmap,
    spotify_genre_comparison,
    spotify_artist_popularity_viz,
    temporal_analysis_plot,
    popularity_vs_feature_scatter,
    create_report_figures
)

# Créer une visualisation de distribution pour une caractéristique
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 6))
spotify_feature_distribution(spotify_df, 'danceability', ax=ax)
plt.show()

# Créer une matrice de corrélation
correlation_fig = spotify_correlation_heatmap(spotify_df)
plt.show()

# Générer toutes les figures d'analyse et les sauvegarder
create_report_figures(spotify_df, "reports/figures")
```

## Script d'analyse automatisée

Le projet inclut un script d'analyse automatisée qui peut être exécuté facilement :

```bash
# Exécuter le script shell qui automatise l'analyse complète
./run_spotify_analysis.sh

# Ou exécuter directement le script Python
python spotify_analysis.py --save-figures --output-dir reports/figures
```

Options disponibles pour le script `spotify_analysis.py` :

```
--output-dir        Répertoire où sauvegarder les figures (défaut: reports/figures)
--save-figures      Activer la sauvegarde des figures générées
--top-artists       Nombre des artistes les plus populaires à analyser (défaut: 15)
--top-genres        Nombre des genres les plus populaires à analyser (défaut: 15)
--no-display        Désactiver l'affichage des figures (utile pour les serveurs)
```

## Exemples d'Analyses Possibles

1. **Analyse par genre musical** : Découvrir quels genres ont les caractéristiques audio les plus distinctives
2. **Facteurs de popularité** : Identifier les caractéristiques audio qui corrèlent le plus avec la popularité
3. **Évolution temporelle** : Observer comment les tendances musicales ont évolué au fil des années
4. **Profils d'artistes** : Comparer les caractéristiques des artistes les plus populaires
5. **Analyse des paroles explicites** : Étudier l'impact du contenu explicite sur la popularité
6. **Clustering des chansons** : Regrouper les chansons selon leurs caractéristiques audio pour découvrir des niches musicales 