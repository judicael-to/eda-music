"""
Module de visualisation pour l'analyse de données Spotify et autres datasets

Ce module fournit des fonctions spécialisées pour créer des visualisations
de haute qualité adaptées à l'analyse des données musicales de Spotify.

Principales fonctions de visualisation:
- spotify_feature_distribution: Visualise la distribution d'une caractéristique audio
- spotify_correlation_heatmap: Crée une matrice de corrélation pour les caractéristiques audio
- spotify_genre_comparison: Compare une caractéristique audio entre différents genres
- spotify_artist_popularity_viz: Visualise les artistes les plus populaires et leurs statistiques
- temporal_analysis_plot: Analyse l'évolution temporelle des caractéristiques audio
- popularity_vs_feature_scatter: Crée un nuage de points entre popularité et caractéristique audio
- create_report_figures: Génère un ensemble complet de visualisations pour un rapport

Chaque fonction est conçue pour être utilisée individuellement ou dans le cadre
d'un flux d'analyse plus large. Le module gère automatiquement les paramètres 
esthétiques pour assurer des visualisations cohérentes et attrayantes.

Utilisation typique:
    from src.data.load_data import load_spotify
    from src.visualization.visualize import spotify_correlation_heatmap
    
    # Charger les données
    df = load_spotify()
    
    # Créer une visualisation
    fig = spotify_correlation_heatmap(df)
    plt.show()
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Any, Tuple
import os
from matplotlib.colors import LinearSegmentedColormap


def setup_aesthetics():
    """Configure l'esthétique générale des visualisations"""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette('viridis')
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12


def spotify_feature_distribution(df: pd.DataFrame, feature: str, ax=None, kde: bool = True) -> plt.Axes:
    """
    Crée un histogramme de distribution pour une caractéristique audio Spotify
    
    Args:
        df: DataFrame pandas contenant les données Spotify
        feature: Nom de la caractéristique à visualiser
        ax: Axes matplotlib (optionnel)
        kde: Ajouter une courbe de densité
    
    Returns:
        Axes matplotlib
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    # Déterminer la couleur en fonction du type de feature
    energy_features = ['energy', 'danceability', 'valence', 'tempo']
    acoustic_features = ['acousticness', 'instrumentalness']
    speech_features = ['speechiness', 'liveness']
    
    if feature in energy_features:
        color = 'orangered'
    elif feature in acoustic_features:
        color = 'forestgreen'
    elif feature in speech_features:
        color = 'royalblue'
    else:
        color = 'mediumpurple'
    
    # Créer l'histogramme
    sns.histplot(df[feature].dropna(), kde=kde, ax=ax, color=color, alpha=0.7)
    
    # Ajouter titre et labels
    ax.set_title(f'Distribution de {feature}', fontsize=14)
    ax.set_xlabel(feature.capitalize())
    ax.set_ylabel('Nombre de chansons')
    
    # Ajouter des lignes pour la moyenne et la médiane
    mean_val = df[feature].mean()
    median_val = df[feature].median()
    
    ax.axvline(mean_val, color='crimson', linestyle='--', linewidth=1.5, label=f'Moyenne: {mean_val:.2f}')
    ax.axvline(median_val, color='navy', linestyle='-.', linewidth=1.5, label=f'Médiane: {median_val:.2f}')
    
    ax.legend()
    
    return ax


def spotify_correlation_heatmap(df: pd.DataFrame, features: List[str] = None, cmap: str = 'coolwarm',
                             figsize: Tuple[int, int] = (12, 10), mask_upper: bool = True) -> plt.Figure:
    """
    Crée une heatmap de corrélation pour les caractéristiques audio Spotify
    
    Args:
        df: DataFrame pandas contenant les données Spotify
        features: Liste des caractéristiques à inclure (par défaut, toutes les caractéristiques audio)
        cmap: Palette de couleurs pour la heatmap
        figsize: Taille de la figure
        mask_upper: Masquer le triangle supérieur de la matrice
    
    Returns:
        Figure matplotlib
    """
    # Si aucune liste de caractéristiques n'est fournie, utiliser les caractéristiques audio par défaut
    if features is None:
        features = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 
            'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo', 'track_popularity', 'artist_popularity'
        ]
    
    # Filtrer pour inclure uniquement les colonnes disponibles
    features = [f for f in features if f in df.columns]
    
    # Calculer la matrice de corrélation
    corr_matrix = df[features].corr()
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Créer un masque pour le triangle supérieur si demandé
    if mask_upper:
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    else:
        mask = None
    
    # Créer la heatmap
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                annot=True, fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": .8},
                ax=ax)
    
    ax.set_title('Matrice de corrélation des caractéristiques audio', fontsize=16, pad=20)
    
    return fig


def spotify_genre_comparison(df: pd.DataFrame, feature: str, top_n: int = 10, fig=None, ax=None) -> plt.Figure:
    """
    Compare une caractéristique audio spécifique entre les genres musicaux
    
    Args:
        df: DataFrame pandas contenant les données Spotify
        feature: Caractéristique audio à comparer
        top_n: Nombre de genres à afficher
        fig: Figure matplotlib (optionnel)
        ax: Axes matplotlib (optionnel)
    
    Returns:
        Figure matplotlib
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Préparer les données - exploser la liste des genres et calculer la moyenne par genre
    df_exploded = df.copy()
    
    # Nettoyer et diviser la colonne de genres
    df_exploded['genres'] = df_exploded['genres'].str.replace('["\[\]]', '', regex=True)
    df_exploded['genres'] = df_exploded['genres'].str.split(', ')
    df_exploded = df_exploded.explode('genres')
    
    # Calculer les top genres par nombre d'occurrences
    top_genres = df_exploded['genres'].value_counts().head(top_n).index
    
    # Filtrer pour ces genres et calculer la moyenne de la caractéristique
    genre_data = df_exploded[df_exploded['genres'].isin(top_genres)]
    genre_means = genre_data.groupby('genres')[feature].mean().sort_values(ascending=False)
    
    # Créer le graphique
    sns.barplot(x=genre_means.values, y=genre_means.index, ax=ax, palette='viridis')
    
    # Ajouter les valeurs sur les barres
    for i, v in enumerate(genre_means.values):
        ax.text(v + 0.01, i, f'{v:.2f}', va='center')
    
    ax.set_title(f'Moyenne de {feature} par genre musical', fontsize=16)
    ax.set_xlabel(feature.capitalize())
    
    return fig


def spotify_artist_popularity_viz(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """
    Visualise les artistes les plus populaires avec leur nombre de followers
    
    Args:
        df: DataFrame pandas contenant les données Spotify
        top_n: Nombre d'artistes à afficher
    
    Returns:
        Figure matplotlib
    """
    # Calculer la popularité moyenne par artiste
    artist_data = df.groupby('artist_name').agg({
        'artist_popularity': 'mean',
        'followers': 'first',
        'track_name': 'count'
    }).reset_index()
    
    # Renommer les colonnes
    artist_data.columns = ['artist_name', 'popularity', 'followers', 'num_tracks']
    
    # Trier par popularité
    top_artists = artist_data.sort_values('popularity', ascending=False).head(top_n)
    
    # Créer une figure avec deux sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Graphique de la popularité
    sns.barplot(y='artist_name', x='popularity', data=top_artists, palette='magma', ax=ax1)
    ax1.set_title(f'Top {top_n} artistes par popularité', fontsize=16)
    ax1.set_xlabel('Popularité')
    ax1.set_ylabel('')
    
    # Ajouter le nombre de titres comme texte
    for i, (_, row) in enumerate(top_artists.iterrows()):
        ax1.text(row['popularity'] + 1, i, f"{row['num_tracks']} titres", va='center')
    
    # Graphique du nombre de followers (échelle logarithmique)
    colors = sns.color_palette('magma', n_colors=len(top_artists))
    bars = ax2.barh(top_artists['artist_name'], top_artists['followers'], color=colors)
    ax2.set_title(f'Nombre de followers par artiste', fontsize=16)
    ax2.set_xlabel('Nombre de followers')
    ax2.set_ylabel('')
    ax2.set_xscale('log')
    
    # Ajouter les valeurs sur les barres
    for bar, followers in zip(bars, top_artists['followers']):
        if followers >= 1_000_000:
            text = f"{followers/1_000_000:.1f}M"
        elif followers >= 1_000:
            text = f"{followers/1_000:.1f}K"
        else:
            text = str(followers)
        
        ax2.text(bar.get_width() * 1.05, bar.get_y() + bar.get_height()/2, text, va='center')
    
    plt.tight_layout()
    
    return fig


def temporal_analysis_plot(df: pd.DataFrame, features: List[str], fig=None, ax=None) -> plt.Figure:
    """
    Crée un graphique d'évolution temporelle des caractéristiques audio
    
    Args:
        df: DataFrame pandas contenant les données Spotify
        features: Liste des caractéristiques audio à inclure
        fig: Figure matplotlib (optionnel)
        ax: Axes matplotlib (optionnel)
    
    Returns:
        Figure matplotlib
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    
    # S'assurer que la date de sortie est au format datetime
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy['release_date']):
        df_copy['release_date'] = pd.to_datetime(df_copy['release_date'])
    
    # Extraire l'année
    df_copy['release_year'] = df_copy['release_date'].dt.year
    
    # Filtrer les années avec trop peu de données
    year_counts = df_copy['release_year'].value_counts()
    valid_years = year_counts[year_counts >= 5].index
    df_filtered = df_copy[df_copy['release_year'].isin(valid_years)]
    
    # Calculer les moyennes annuelles pour chaque caractéristique
    yearly_data = df_filtered.groupby('release_year')[features].mean().reset_index()
    
    # Créer le graphique d'évolution
    colors = plt.cm.tab10(np.linspace(0, 1, len(features)))
    
    for i, feature in enumerate(features):
        ax.plot(yearly_data['release_year'], yearly_data[feature], 
                marker='o', linestyle='-', color=colors[i], linewidth=2, label=feature)
    
    ax.set_title('Évolution des caractéristiques audio au fil des années', fontsize=16)
    ax.set_xlabel('Année')
    ax.set_ylabel('Valeur moyenne')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)
    
    # Ajuster les marges pour la légende
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)
    
    return fig


def popularity_vs_feature_scatter(df: pd.DataFrame, feature: str, color_by: str = None, fig=None, ax=None) -> plt.Figure:
    """
    Crée un scatter plot de la popularité vs une caractéristique audio
    
    Args:
        df: DataFrame pandas contenant les données Spotify
        feature: Caractéristique audio à mettre en relation avec la popularité
        color_by: Colonne à utiliser pour colorer les points (optionnel)
        fig: Figure matplotlib (optionnel)
        ax: Axes matplotlib (optionnel)
    
    Returns:
        Figure matplotlib
    """
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    
    # Paramètres de base du scatter plot
    plot_params = {
        'x': feature,
        'y': 'track_popularity',
        'data': df,
        'alpha': 0.6,
        'ax': ax
    }
    
    # Ajouter la coloration si spécifiée
    if color_by and color_by in df.columns:
        if df[color_by].dtype in ['int64', 'float64']:
            scatter = ax.scatter(
                df[feature], 
                df['track_popularity'],
                c=df[color_by], 
                cmap='viridis',
                alpha=0.6
            )
            plt.colorbar(scatter, ax=ax, label=color_by)
        else:
            sns.scatterplot(**plot_params, hue=color_by)
    else:
        sns.scatterplot(**plot_params)
    
    # Ajouter une ligne de régression
    sns.regplot(x=feature, y='track_popularity', data=df, scatter=False, 
                ax=ax, color='red', line_kws={'linestyle':'--'})
    
    # Calculer et afficher le coefficient de corrélation
    corr = df[[feature, 'track_popularity']].corr().iloc[0, 1]
    ax.text(0.05, 0.95, f'Corrélation: {corr:.2f}', transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))
    
    ax.set_title(f'Relation entre {feature} et popularité des titres', fontsize=16)
    ax.set_xlabel(feature.capitalize())
    ax.set_ylabel('Popularité du titre')
    ax.grid(True, alpha=0.3)
    
    return fig


def create_report_figures(df: pd.DataFrame, output_dir: str) -> None:
    """
    Crée et sauvegarde un ensemble de figures pour un rapport d'analyse Spotify
    
    Args:
        df: DataFrame pandas contenant les données Spotify
        output_dir: Répertoire où sauvegarder les figures
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)
    
    # Configurer l'esthétique
    setup_aesthetics()
    
    # 1. Matrice de corrélation
    fig1 = spotify_correlation_heatmap(df)
    fig1.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Top artistes
    fig2 = spotify_artist_popularity_viz(df, top_n=10)
    fig2.savefig(os.path.join(output_dir, 'top_artists.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Distribution des caractéristiques audio principales
    fig3, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    features = ['danceability', 'energy', 'valence', 'acousticness']
    for i, feature in enumerate(features):
        spotify_feature_distribution(df, feature, ax=axes[i])
    
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'audio_features_dist.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Comparaison par genre
    fig4 = plt.figure(figsize=(12, 8))
    spotify_genre_comparison(df, 'energy', top_n=10, fig=fig4, ax=fig4.gca())
    fig4.savefig(os.path.join(output_dir, 'genre_energy.png'), dpi=300, bbox_inches='tight')
    plt.close(fig4)
    
    # 5. Évolution temporelle
    fig5 = plt.figure(figsize=(14, 8))
    temporal_analysis_plot(df, ['danceability', 'energy', 'valence', 'acousticness'], fig=fig5, ax=fig5.gca())
    fig5.savefig(os.path.join(output_dir, 'temporal_trends.png'), dpi=300, bbox_inches='tight')
    plt.close(fig5)
    
    # 6. Popularité vs caractéristiques
    fig6, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(['danceability', 'energy', 'acousticness', 'valence']):
        popularity_vs_feature_scatter(df, feature, ax=axes[i])
    
    plt.tight_layout()
    fig6.savefig(os.path.join(output_dir, 'popularity_vs_features.png'), dpi=300, bbox_inches='tight')
    plt.close(fig6)
    
    print(f"Figures sauvegardées dans {output_dir}")


# Si le script est exécuté directement
if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    
    # Ajouter le répertoire parent au path pour pouvoir importer depuis src
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    
    from src.data.load_data import load_spotify
    
    parser = argparse.ArgumentParser(description="Générer des visualisations pour le dataset Spotify")
    parser.add_argument('--output', type=str, default='reports/figures', 
                        help='Répertoire où sauvegarder les figures')
    
    args = parser.parse_args()
    
    # Charger le dataset Spotify
    spotify_df = load_spotify()
    
    # Créer les figures
    create_report_figures(spotify_df, args.output)
