# Rapports d'Analyse

Ce répertoire contient les rapports générés par le framework d'analyse exploratoire des données.

## Structure

- `figures/` : Visualisations générées par les analyses
- `*.html` : Rapports HTML exportés depuis les notebooks Jupyter
- `*.pdf` : Rapports PDF (si générés)

## Génération des rapports

Pour générer un rapport complet d'analyse du dataset Spotify, exécutez le script suivant depuis la racine du projet :

```bash
./run_spotify_analysis.sh
```

Pour générer uniquement les figures :

```bash
python spotify_analysis.py --save-figures --output-dir reports/figures
```

## Visualisation des rapports

Les rapports HTML peuvent être ouverts directement dans un navigateur web. Les figures individuelles sont disponibles dans le dossier `figures/`. 