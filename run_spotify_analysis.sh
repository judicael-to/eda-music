#!/bin/bash

# Script pour exécuter l'analyse complète du dataset Spotify et générer des rapports
# Ce script automatise l'ensemble du flux de travail d'analyse en:
# 1. Vérifiant la présence des fichiers nécessaires
# 2. Exécutant l'analyse Python et générant les visualisations
# 3. Produisant un rapport HTML à partir du notebook Jupyter

# Créer les répertoires de sortie s'ils n'existent pas
mkdir -p reports/figures

echo "===== ANALYSE DU DATASET SPOTIFY ====="

# Installer le package en mode développement si nécessaire
# Utilisez --install comme argument pour installer le package
if [ "$1" == "--install" ]; then
    echo "Installation du package en mode développement..."
    pip install -e .
    echo "Package installé avec succès."
fi

# Vérifier la présence du dataset Spotify
if [ ! -f "spotifydataset.csv" ]; then
    echo "Erreur: Le fichier spotifydataset.csv est introuvable!"
    echo "Veuillez placer le fichier à la racine du projet."
    exit 1
fi

# Vérifier la présence du notebook d'analyse
if [ ! -f "notebooks/spotify_eda.ipynb" ]; then
    echo "Avertissement: Le notebook spotify_eda.ipynb est introuvable."
    echo "Le rapport HTML ne sera pas généré."
    SKIP_NOTEBOOK=1
fi

# Exécuter l'analyse Python et générer les visualisations
echo "Génération des visualisations et analyse des données..."
python ./spotify_analysis.py --save-figures --output-dir reports/figures --no-display

# Vérifier si l'analyse Python s'est terminée correctement
if [ $? -ne 0 ]; then
    echo "Erreur lors de l'exécution de l'analyse Python."
    exit 1
fi

# Générer un rapport HTML à partir du notebook Jupyter (si disponible)
if [ -z "$SKIP_NOTEBOOK" ]; then
    echo "Génération du rapport HTML à partir du notebook..."
    jupyter nbconvert --execute --to html notebooks/spotify_eda.ipynb --output ../reports/spotify_analysis_report.html
    
    # Vérifier si la conversion du notebook s'est terminée correctement
    if [ $? -ne 0 ]; then
        echo "Avertissement: Problème lors de la génération du rapport HTML."
    else
        echo "Rapport HTML généré avec succès: reports/spotify_analysis_report.html"
    fi
fi

echo "===== ANALYSE TERMINÉE ====="
echo "Rapports générés dans le répertoire 'reports/'"
echo "Visualisations générées dans le répertoire 'reports/figures/'"

# Informations supplémentaires sur l'utilisation
echo
echo "Pour visualiser les résultats:"
echo "- Ouvrez le rapport HTML: reports/spotify_analysis_report.html (si généré)"
echo "- Consultez les visualisations dans: reports/figures/"
echo
echo "Pour une analyse interactive, exécutez:"
echo "jupyter notebook notebooks/spotify_eda.ipynb" 