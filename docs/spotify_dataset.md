# Documentation du Dataset Spotify

## Description

Ce dataset contient des informations sur des artistes, des chansons, et leurs caractéristiques audio sur Spotify. Il permet d'analyser les tendances musicales, les caractéristiques des genres et les facteurs qui influencent la popularité.

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

Notre framework d'analyse exploratoire permet d'analyser facilement ce dataset avec des fonctions dédiées :

```python
from src.data.load_data import load_spotify
from src.analysis.spotify_analysis import analyser_spotify_dataset

# Charger le dataset
spotify_df = load_spotify()

# Lancer l'analyse complète
analyser_spotify_dataset(spotify_df)
```

Pour une analyse visuelle plus poussée:

```python
from src.visualization.visualize import create_report_figures

# Générer toutes les figures d'analyse
create_report_figures(spotify_df, "reports/figures")
```

## Exemples d'Analyses Possibles

1. **Analyse par genre musical** : Découvrir quels genres ont les caractéristiques audio les plus distinctives
2. **Facteurs de popularité** : Identifier les caractéristiques audio qui corrèlent le plus avec la popularité
3. **Évolution temporelle** : Observer comment les tendances musicales ont évolué au fil des années
4. **Profils d'artistes** : Comparer les caractéristiques des artistes les plus populaires
5. **Analyse des paroles explicites** : Étudier l'impact du contenu explicite sur la popularité 