# Visualisations du Dataset Spotify

Ce répertoire contient les visualisations générées par l'analyse du dataset Spotify.

## Contenu des visualisations

- `correlation_matrix.png` : Matrice de corrélation des caractéristiques audio
- `top_artists.png` : Top artistes les plus populaires avec leur nombre de followers
- `audio_features_dist.png` : Distribution des caractéristiques audio principales
- `genre_energy.png` : Comparaison de l'énergie par genre musical
- `temporal_trends.png` : Évolution des caractéristiques audio au fil des années
- `popularity_vs_features.png` : Relation entre caractéristiques audio et popularité

## Génération des visualisations

Pour générer ces visualisations, exécutez :

```bash
python spotify_analysis.py --save-figures --output-dir reports/figures
```

Ou pour une analyse complète avec génération de rapport :

```bash
./run_spotify_analysis.sh
``` 