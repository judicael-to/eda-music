"""
Module d'analyse spécifique au dataset Spotify

Ce module contient des fonctions spécialisées pour l'analyse des données musicales
provenant de Spotify. Il permet d'explorer les caractéristiques audio, les relations
entre genres musicaux, les facteurs de popularité, et les tendances temporelles.

Fonctions principales:
- analyser_correlations_audio: Analyse les corrélations entre métriques audio
- analyser_par_genre: Analyse les caractéristiques audio par genre musical
- analyser_popularite: Analyse les facteurs liés à la popularité des artistes/chansons
- analyser_distribution_audio: Analyse la distribution des caractéristiques audio
- analyser_tendances_temporelles: Analyse l'évolution des caractéristiques dans le temps
- analyser_top_artistes: Identifie et analyse les artistes les plus populaires
- analyser_spotify_dataset: Point d'entrée principal pour l'analyse complète

Utilisation:
    from src.data.load_data import load_spotify
    from src.analysis.spotify_analysis import analyser_spotify_dataset
    
    # Charger les données
    df = load_spotify()
    
    # Exécuter l'analyse complète
    analyser_spotify_dataset(df)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
import os

# Analyse des corrélations entre les métriques audio
def analyser_correlations_audio(df: pd.DataFrame) -> list:
    """
    Analyse les corrélations entre les différentes métriques audio
    
    Args:
        df: DataFrame pandas contenant les données Spotify
        
    Returns:
        list: Liste des figures générées
    """
    figures = []
    
    # Sélectionner uniquement les caractéristiques audio
    audio_features = [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
        'track_popularity'
    ]
    
    # Matrices de corrélation
    corr_matrix = df[audio_features].corr()
    
    # Visualiser la matrice de corrélation
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Matrice de corrélation des caractéristiques audio', fontsize=16)
    plt.tight_layout()
    figures.append(fig)
    
    return figures


# Analyse par genre musical
def analyser_par_genre(df: pd.DataFrame, n_genres: int = 15) -> list:
    """
    Analyse les caractéristiques audio par genre musical
    
    Args:
        df: DataFrame pandas contenant les données Spotify
        n_genres: Nombre de genres les plus populaires à analyser
    
    Returns:
        list: Liste des figures générées
        pd.DataFrame: DataFrame avec les moyennes des caractéristiques par genre
    """
    figures = []
    
    # Exploser le champ genres qui contient plusieurs genres par ligne
    genres_exploded = df.copy()
    
    # Nettoyer et diviser la colonne de genres
    genres_exploded['genres'] = genres_exploded['genres'].str.replace('[\"\\[\\]]', '', regex=True)
    genres_exploded['genres'] = genres_exploded['genres'].str.split(', ')
    genres_exploded = genres_exploded.explode('genres')
    
    # Compter le nombre de titres par genre
    genre_counts = genres_exploded['genres'].value_counts().reset_index()
    genre_counts.columns = ['genre', 'count']
    
    # Sélectionner les n_genres les plus populaires
    top_genres = genre_counts.head(n_genres)
    
    # Visualiser les genres les plus populaires
    fig1 = plt.figure(figsize=(12, 6))
    sns.barplot(x='count', y='genre', data=top_genres)
    plt.title(f'Les {n_genres} genres les plus populaires', fontsize=14)
    plt.xlabel('Nombre de titres')
    plt.ylabel('Genre')
    plt.tight_layout()
    figures.append(fig1)
    
    # Analyser les caractéristiques audio par genre
    audio_features = [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    # Filtrer pour ne garder que les genres les plus populaires
    top_genres_list = top_genres['genre'].tolist()
    genres_filtered = genres_exploded[genres_exploded['genres'].isin(top_genres_list)]
    
    # Calculer la moyenne des caractéristiques audio par genre
    genre_means = genres_filtered.groupby('genres')[audio_features].mean().reset_index()
    
    # Créer un radar chart pour comparer les genres
    # Sélectionner quelques genres représentatifs pour une meilleure lisibilité
    sample_genres = genre_means.iloc[:5]
    
    # Préparer les données pour le radar chart
    categories = audio_features
    N = len(categories)
    
    # Créer le radar chart
    fig2 = plt.figure(figsize=(12, 10))
    
    # Calculer les angles pour chaque caractéristique
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Fermer le cercle
    
    # Initialiser le subplot en tant que polar plot
    ax = plt.subplot(111, polar=True)
    
    # Ajouter les caractéristiques comme des rayons du cercle
    plt.xticks(angles[:-1], categories, size=12)
    
    # Dessiner le radar pour chaque genre
    for i, row in sample_genres.iterrows():
        values = row[audio_features].values.flatten().tolist()
        values += values[:1]  # Fermer le cercle
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=row['genres'])
        ax.fill(angles, values, alpha=0.1)
    
    plt.title('Comparaison des caractéristiques audio par genre', size=16)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    figures.append(fig2)
    
    return figures, genre_means


# Analyse de popularité des artistes et chansons
def analyser_popularite(df: pd.DataFrame) -> list:
    """
    Analyse les facteurs liés à la popularité des artistes et des chansons
    
    Args:
        df: DataFrame pandas contenant les données Spotify
        
    Returns:
        list: Liste des figures générées
    """
    figures = []
    
    # Corrélation entre popularité de l'artiste et nombre de followers
    fig1 = plt.figure(figsize=(10, 6))
    sns.scatterplot(x='followers', y='artist_popularity', data=df)
    plt.title('Relation entre nombre de followers et popularité de l\'artiste', fontsize=14)
    plt.xlabel('Nombre de followers')
    plt.ylabel('Popularité de l\'artiste')
    plt.xscale('log')  # Échelle logarithmique pour mieux visualiser
    plt.tight_layout()
    figures.append(fig1)
    
    # Corrélation entre popularité de la chanson et caractéristiques audio
    audio_features = [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    # Calculer les corrélations avec la popularité des chansons
    correlations = df[audio_features + ['track_popularity']].corr()['track_popularity'].drop('track_popularity').sort_values(ascending=False)
    
    fig2 = plt.figure(figsize=(10, 6))
    sns.barplot(x=correlations.values, y=correlations.index)
    plt.title('Corrélation entre caractéristiques audio et popularité des chansons', fontsize=14)
    plt.xlabel('Coefficient de corrélation')
    plt.tight_layout()
    figures.append(fig2)
    
    # Impact de l'aspect explicite sur la popularité
    fig3 = plt.figure(figsize=(10, 6))
    sns.boxplot(x='explicit', y='track_popularity', data=df)
    plt.title('Impact du contenu explicite sur la popularité des chansons', fontsize=14)
    plt.xlabel('Contenu explicite')
    plt.ylabel('Popularité de la chanson')
    plt.tight_layout()
    figures.append(fig3)
    
    return figures


# Analyse de la distribution des caractéristiques audio
def analyser_distribution_audio(df: pd.DataFrame) -> list:
    """
    Analyse la distribution des caractéristiques audio des chansons
    
    Args:
        df: DataFrame pandas contenant les données Spotify
        
    Returns:
        list: Liste des figures générées
    """
    figures = []
    
    # Caractéristiques audio à analyser
    audio_features = [
        'danceability', 'energy', 'loudness', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    
    # Créer un grid de histogrammes
    fig1, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, feature in enumerate(audio_features):
        sns.histplot(df[feature], kde=True, ax=axes[i])
        axes[i].set_title(f'Distribution de {feature}', fontsize=12)
    
    plt.tight_layout()
    figures.append(fig1)
    
    # Créer un pairplot pour voir les relations entre certaines caractéristiques clés
    key_features = ['danceability', 'energy', 'valence', 'tempo']
    g = sns.pairplot(df[key_features + ['track_popularity']], 
                 palette='viridis',
                 plot_kws={'alpha': 0.5, 's': 15})
    g.fig.suptitle('Relations entre caractéristiques audio et popularité', y=1.02, fontsize=16)
    plt.tight_layout()
    figures.append(g.fig)
    
    return figures


# Analyse temporelle des sorties de titres
def analyser_tendances_temporelles(df: pd.DataFrame) -> list:
    """
    Analyse les tendances temporelles dans les sorties de titres
    
    Args:
        df: DataFrame pandas contenant les données Spotify
        
    Returns:
        list: Liste des figures générées
    """
    figures = []
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Handle different date formats by checking the length
    # If it's just a year, add a default month and day
    df_copy['release_year'] = df_copy['release_date'].apply(
        lambda x: int(x[:4]) if isinstance(x, str) else None
    )
    
    # Analyze only rows with valid years
    df_copy = df_copy.dropna(subset=['release_year'])
    
    # List of features to analyze
    features = ['danceability', 'energy', 'acousticness', 'valence']
    
    # Analyser l'évolution des caractéristiques audio au fil des années
    yearly_features = df_copy.groupby('release_year')[features].mean().reset_index()
    
    # Visualiser l'évolution
    fig1 = plt.figure(figsize=(12, 8))
    
    for feature in features:
        plt.plot(yearly_features['release_year'], yearly_features[feature], marker='o', label=feature)
    
    plt.title('Évolution des caractéristiques audio au fil des années', fontsize=14)
    plt.xlabel('Année de sortie')
    plt.ylabel('Valeur moyenne')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    figures.append(fig1)
    
    # Nombre de sorties par année
    releases_per_year = df_copy['release_year'].value_counts().sort_index()
    
    fig2 = plt.figure(figsize=(12, 6))
    sns.barplot(x=releases_per_year.index, y=releases_per_year.values)
    plt.title('Nombre de titres par année dans le dataset', fontsize=14)
    plt.xlabel('Année')
    plt.ylabel('Nombre de titres')
    plt.xticks(rotation=45)
    plt.tight_layout()
    figures.append(fig2)
    
    return figures


# Fonction principale pour exécuter toutes les analyses
def analyser_spotify_dataset(df: pd.DataFrame) -> None:
    """
    Exécute l'ensemble des analyses sur le dataset Spotify
    
    Args:
        df: DataFrame pandas contenant les données Spotify
    """
    print("=== ANALYSE DU DATASET SPOTIFY ===\n")
    
    # Aperçu des données
    print(f"Dimensions du dataset: {df.shape[0]} lignes × {df.shape[1]} colonnes\n")
    print("Premières lignes du dataset:")
    print(df.head())
    
    print("\nStatistiques descriptives:")
    print(df.describe().T)
    
    # Exécuter les analyses spécifiques
    print("\n=== ANALYSE DES CORRÉLATIONS ENTRE MÉTRIQUES AUDIO ===")
    figures = analyser_correlations_audio(df)
    
    print("\n=== ANALYSE PAR GENRE MUSICAL ===")
    figures, genre_means = analyser_par_genre(df)
    print("\nMoyenne des caractéristiques par genre:")
    print(genre_means)
    
    print("\n=== ANALYSE DE LA POPULARITÉ ===")
    figures = analyser_popularite(df)
    
    print("\n=== DISTRIBUTION DES CARACTÉRISTIQUES AUDIO ===")
    figures = analyser_distribution_audio(df)
    
    print("\n=== TENDANCES TEMPORELLES ===")
    figures = analyser_tendances_temporelles(df)
    
    print("\n=== FIN DE L'ANALYSE ===")


# Analyse par artiste
def analyser_top_artistes(df: pd.DataFrame, n_artistes: int = 10) -> list:
    """
    Analyse les artistes les plus populaires du dataset
    
    Args:
        df: DataFrame pandas contenant les données Spotify
        n_artistes: Nombre d'artistes les plus populaires à analyser
        
    Returns:
        list: Liste des figures générées
    """
    figures = []
    
    # Regrouper par artiste et calculer la moyenne de popularité et le nombre de titres
    artist_data = df.groupby('artist_name').agg({
        'artist_popularity': 'mean',
        'track_popularity': 'mean',
        'followers': 'first',
        'track_name': 'count'
    }).reset_index()
    
    artist_data.columns = ['artist_name', 'artist_popularity', 'avg_track_popularity', 'followers', 'track_count']
    
    # Trier par popularité de l'artiste
    top_artists = artist_data.sort_values('artist_popularity', ascending=False).head(n_artistes)
    
    # Visualiser les résultats
    fig1 = plt.figure(figsize=(12, 6))
    sns.barplot(x='artist_popularity', y='artist_name', data=top_artists)
    plt.title(f'Top {n_artistes} des artistes les plus populaires', fontsize=14)
    plt.xlabel('Popularité')
    plt.ylabel('Artiste')
    plt.tight_layout()
    figures.append(fig1)
    
    # Comparer popularité de l'artiste vs popularité moyenne des titres
    fig2 = plt.figure(figsize=(12, 6))
    
    x = top_artists['artist_popularity']
    y = top_artists['avg_track_popularity']
    sizes = top_artists['followers'] / 1e6  # Taille des points proportionnelle au nombre de followers (en millions)
    
    # Use a colormap to map sizes to colors for better visualization
    norm = plt.Normalize(sizes.min(), sizes.max())
    scatter = plt.scatter(x, y, s=sizes*20, alpha=0.7, c=sizes, cmap='viridis', norm=norm)
    
    # Ajouter les noms des artistes
    for i, artist in enumerate(top_artists['artist_name']):
        plt.annotate(artist, (x.iloc[i], y.iloc[i]), fontsize=9)
    
    plt.colorbar(scatter, label='Millions de followers')
    plt.xlabel('Popularité de l\'artiste')
    plt.ylabel('Popularité moyenne des titres')
    plt.title('Comparaison entre popularité de l\'artiste et de ses titres', fontsize=14)
    plt.tight_layout()
    figures.append(fig2)
    
    # Nombre moyen de titres par artiste
    fig3 = plt.figure(figsize=(10, 6))
    sns.barplot(x='track_count', y='artist_name', data=top_artists)
    plt.title('Nombre de titres par artiste dans le dataset', fontsize=14)
    plt.xlabel('Nombre de titres')
    plt.ylabel('Artiste')
    plt.tight_layout()
    figures.append(fig3)
    
    print(top_artists)
    
    return figures 