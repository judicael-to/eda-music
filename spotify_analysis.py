#!/usr/bin/env python3
"""
Script pour exécuter l'analyse du dataset Spotify

Ce script est l'interface principale pour l'analyse des données Spotify. Il permet d'effectuer 
une analyse complète des caractéristiques audio, artistes, genres et tendances dans le dataset.

Fonctionnalités:
- Analyse des corrélations entre caractéristiques audio (danceability, energy, etc.)
- Analyse des genres musicaux et leurs caractéristiques distinctives
- Analyse des facteurs de popularité des artistes et des chansons
- Analyse des distributions des caractéristiques audio
- Analyse des tendances temporelles dans les données musicales
- Visualisation des artistes les plus populaires

Utilisation:
    python spotify_analysis.py [options]

Options:
    --output-dir DIR     Répertoire où sauvegarder les figures (défaut: reports/figures)
    --save-figures       Activer la sauvegarde des figures générées
    --top-artists N      Nombre des artistes les plus populaires à analyser (défaut: 15)
    --top-genres N       Nombre des genres les plus populaires à analyser (défaut: 15)
    --no-display         Désactiver l'affichage des figures (utile pour les environnements sans interface graphique)

Exemples:
    # Exécuter l'analyse complète avec affichage des figures
    python spotify_analysis.py
    
    # Exécuter l'analyse et sauvegarder les figures dans un répertoire personnalisé
    python spotify_analysis.py --save-figures --output-dir mes_resultats/figures
    
    # Exécuter l'analyse en mode serveur (sans affichage)
    python spotify_analysis.py --no-display --save-figures
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys

# Ajouter le répertoire parent au path pour pouvoir importer depuis src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Point d'entrée principal du script
    """
    # Configurer l'analyseur d'arguments
    parser = argparse.ArgumentParser(description="Analyse du dataset Spotify")
    
    parser.add_argument('--output-dir', type=str, default='reports/figures',
                        help='Répertoire où sauvegarder les figures générées')
    
    parser.add_argument('--save-figures', action='store_true',
                        help='Activer la sauvegarde des figures générées')
    
    parser.add_argument('--top-artists', type=int, default=15,
                        help='Nombre des artistes les plus populaires à analyser')
    
    parser.add_argument('--top-genres', type=int, default=15,
                        help='Nombre des genres les plus populaires à analyser')
    
    parser.add_argument('--no-display', action='store_true',
                        help='Désactiver l\'affichage des figures (utile pour les environnements sans interface graphique)')
    
    args = parser.parse_args()
    
    # Si la sauvegarde est activée, créer le répertoire de sortie
    if args.save_figures:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Les figures seront sauvegardées dans {args.output_dir}")
    
    # Configuration pour l'affichage ou non des figures
    if args.no_display:
        plt.ioff()  # Désactiver l'affichage interactif
    else:
        plt.ion()   # Activer l'affichage interactif
    
    # Configuration pour une meilleure visualisation
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('viridis')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    # Importer les fonctions d'analyse
    from src.data.load_data import load_spotify
    from src.analysis.spotify_analysis import (
        analyser_spotify_dataset, 
        analyser_top_artistes, 
        analyser_par_genre,
        analyser_correlations_audio,
        analyser_popularite,
        analyser_distribution_audio,
        analyser_tendances_temporelles
    )
    
    # Charger le dataset Spotify
    print("Chargement du dataset Spotify...")
    spotify_df = load_spotify()
    print(f"Dataset chargé : {spotify_df.shape[0]} lignes × {spotify_df.shape[1]} colonnes")
    
    # Fonction pour sauvegarder les figures si nécessaire
    def save_fig(figure, name, index=""):
        if args.save_figures:
            # Make sure it's a valid figure
            if figure.get_axes():
                fig_path = os.path.join(args.output_dir, f"{name}{index}.png")
                figure.savefig(fig_path, dpi=300, bbox_inches='tight')
                print(f"Figure sauvegardée : {fig_path}")
    
    def save_figures(figures_list, base_name):
        if not isinstance(figures_list, list):  # If it's a single figure
            figures_list = [figures_list]
        
        for i, fig in enumerate(figures_list):
            index = f"_{i+1}" if len(figures_list) > 1 else ""
            save_fig(fig, base_name, index)
            if not args.no_display:
                plt.figure(fig.number)
                plt.show(block=False)
    
    # Analyser les top artistes
    print("\nAnalyse des artistes les plus populaires...")
    fig_top_artistes = analyser_top_artistes(spotify_df, n_artistes=args.top_artists)
    save_figures(fig_top_artistes, "top_artistes")
    
    # Analyser les corrélations audio
    print("\nAnalyse des corrélations entre métriques audio...")
    fig_correlations = analyser_correlations_audio(spotify_df)
    save_figures(fig_correlations, "correlations_audio")
    
    # Analyser par genre
    print("\nAnalyse par genre musical...")
    result_genres = analyser_par_genre(spotify_df, n_genres=args.top_genres)
    fig_genres = result_genres[0]  # First element is the figures list
    genre_means = result_genres[1]  # Second element is the genre_means DataFrame
    save_figures(fig_genres, "analyse_genres")
    
    # Analyser la popularité
    print("\nAnalyse des facteurs de popularité...")
    figs_popularite = analyser_popularite(spotify_df)
    save_figures(figs_popularite, "facteurs_popularite")
    
    # Analyser la distribution des caractéristiques audio
    print("\nAnalyse de la distribution des caractéristiques audio...")
    figs_distribution = analyser_distribution_audio(spotify_df)
    save_figures(figs_distribution, "distribution_audio")
    
    # Analyser les tendances temporelles
    print("\nAnalyse des tendances temporelles...")
    figs_tendances = analyser_tendances_temporelles(spotify_df)
    save_figures(figs_tendances, "tendances_temporelles")
    
    print("\nAnalyse terminée avec succès!")
    
    # Si l'affichage est désactivé mais que les figures sont générées
    if args.no_display and args.save_figures:
        print(f"Les figures ont été sauvegardées dans {args.output_dir}")
    elif not args.no_display:
        print("Fermer les fenêtres de figures pour terminer.")
        plt.show(block=True)  # Attendre que l'utilisateur ferme les fenêtres


if __name__ == "__main__":
    main() 