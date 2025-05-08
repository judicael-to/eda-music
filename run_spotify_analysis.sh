#!/bin/bash

# Script pour exécuter l'analyse complète du dataset Spotify et générer des rapports

# Créer les répertoires de sortie s'ils n'existent pas
mkdir -p reports/figures

echo "===== ANALYSE DU DATASET SPOTIFY ====="

# Installer le package en mode développement si nécessaire
if [ "$1" == "--install" ]; then
    echo "Installation du package..."
    pip install -e .
fi

# Vérifier la présence du dataset
if [ ! -f "spotifydataset.csv" ]; then
    echo "Erreur: Le fichier spotifydataset.csv est introuvable!"
    exit 1
fi

# Exécuter l'analyse et générer des figures
echo "Génération des visualisations..."
python ./spotify_analysis.py --save-figures --output-dir reports/figures

# Exécuter le notebook pour générer un rapport HTML
echo "Génération du rapport HTML à partir du notebook..."
jupyter nbconvert --execute --to html notebooks/spotify_eda.ipynb --output ../reports/spotify_analysis_report.html

echo "===== ANALYSE TERMINÉE ====="
echo "Rapports générés dans le répertoire 'reports/'"
echo "Visualisations générées dans le répertoire 'reports/figures/'" 