"""
Module principal pour l'analyse exploratoire de données
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno
import warnings
from IPython.display import display, HTML
from typing import List, Optional, Dict, Any, Union
import os

# Ignorer les avertissements pour une sortie plus propre
warnings.filterwarnings('ignore')

# Configurer le style de visualisation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Fonction principale pour l'analyse exploratoire
def analyse_exploratoire(df: pd.DataFrame, titre: str = "Analyse Exploratoire de Données", 
                        output_dir: Optional[str] = None):
    """
    Réalise une analyse exploratoire complète sur un dataframe
    
    Args:
        df: DataFrame pandas à analyser
        titre: Titre de l'analyse
        output_dir: Répertoire où sauvegarder les figures générées (optionnel)
    """
    print(f"==== {titre} ====\n")
    
    # Créer le répertoire de sortie si spécifié
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 1. Aperçu des données
    apercu_donnees(df)
    
    # 2. Analyse des valeurs manquantes
    analyse_valeurs_manquantes(df, output_dir)
    
    # 3. Analyse univariée
    analyse_univariee(df, output_dir)
    
    # 4. Analyse bivariée
    analyse_bivariee(df, output_dir)
    
    # 5. Analyse des corrélations
    analyse_correlations(df, output_dir)
    
    # 6. Distribution des variables numériques
    distribution_variables_numeriques(df, output_dir)
    
    # 7. Analyse temporelle (si applicable)
    colonnes_date = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
    if not colonnes_date:
        # Essayer de détecter les colonnes de date au format string
        date_candidates = [col for col in df.select_dtypes(include=['object']).columns 
                         if col.lower().find('date') >= 0 or col.lower().find('time') >= 0]
        
        if date_candidates:
            print(f"\nColonnes potentiellement temporelles détectées: {date_candidates}")
            print("Vous pouvez convertir ces colonnes en datetime pour une analyse temporelle.")
    else:
        for col in colonnes_date:
            analyse_temporelle(df, col, output_dir)
    
    # 8. Détection spécifique du dataset Spotify
    if is_spotify_dataset(df):
        print("\n=== DATASET SPOTIFY DÉTECTÉ ===")
        print("Pour une analyse complète du dataset Spotify, utilisez le module spotify_analysis:")
        print("from src.analysis.spotify_analysis import analyser_spotify_dataset")
        print("analyser_spotify_dataset(df)")
    
    print("\n==== Fin de l'analyse exploratoire ====")


def is_spotify_dataset(df: pd.DataFrame) -> bool:
    """
    Détecte si le dataframe est probablement le dataset Spotify
    
    Args:
        df: DataFrame pandas à analyser
    
    Returns:
        True si le dataset ressemble au dataset Spotify, False sinon
    """
    # Vérifier si les colonnes typiques du dataset Spotify sont présentes
    spotify_cols = ['artist_name', 'track_name', 'danceability', 'energy', 'valence', 
                   'tempo', 'album_name', 'artist_popularity']
    
    # Compter combien de colonnes typiques sont présentes
    matches = sum(1 for col in spotify_cols if col in df.columns)
    
    # Si au moins 5 colonnes correspondent, c'est probablement le dataset Spotify
    return matches >= 5


def apercu_donnees(df: pd.DataFrame) -> None:
    """Affiche un aperçu général des données"""
    print("=== APERÇU DES DONNÉES ===")
    print(f"\nDimensions du dataset: {df.shape[0]} lignes × {df.shape[1]} colonnes")
    
    print("\n• Premières lignes du dataset:")
    display(df.head())
    
    print("\n• Informations sur les types de données:")
    buffer = []
    for dtype in df.dtypes.value_counts().items():
        buffer.append(f"   - {dtype[0]}: {dtype[1]} colonnes")
    print("\n".join(buffer))
    
    print("\n• Description des colonnes:")
    display(df.dtypes)
    
    print("\n• Statistiques descriptives:")
    display(df.describe().T)
    
    # Afficher les informations sur les variables catégorielles
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        print("\n• Variables catégorielles:")
        for col in cat_cols:
            print(f"\n   - {col}: {df[col].nunique()} valeurs uniques")
            if df[col].nunique() < 15:  # Si moins de 15 valeurs uniques
                print(f"     Valeurs: {', '.join(df[col].value_counts().nlargest(10).index.astype(str))}")
                print(f"     Distribution: {', '.join([f'{v}:{c}' for v, c in zip(df[col].value_counts().nlargest(5).index, df[col].value_counts().nlargest(5).values)])}")


def analyse_valeurs_manquantes(df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """
    Analyse et visualise les valeurs manquantes
    
    Args:
        df: DataFrame pandas à analyser
        output_dir: Répertoire où sauvegarder les figures générées (optionnel)
    """
    print("\n=== ANALYSE DES VALEURS MANQUANTES ===")
    
    # Calculer le nombre et pourcentage de valeurs manquantes
    missing_values = df.isnull().sum()
    missing_values_percent = 100 * missing_values / len(df)
    missing_df = pd.DataFrame({
        'Valeurs manquantes': missing_values,
        'Pourcentage (%)': missing_values_percent.round(2)
    }).sort_values('Valeurs manquantes', ascending=False)
    
    # Afficher les colonnes avec des valeurs manquantes
    missing_df_filtered = missing_df[missing_df['Valeurs manquantes'] > 0]
    
    if len(missing_df_filtered) > 0:
        print("\n• Colonnes avec valeurs manquantes:")
        display(missing_df_filtered)
        
        # Visualisation des valeurs manquantes
        plt.figure(figsize=(12, 6))
        plt.title('Pourcentage de valeurs manquantes par colonne')
        sns.barplot(x=missing_df_filtered.index, y='Pourcentage (%)', data=missing_df_filtered)
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'valeurs_manquantes_barplot.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Visualisation de la matrice des valeurs manquantes
        try:
            plt.figure(figsize=(12, 8))
            msno.matrix(df)
            plt.title('Matrice des valeurs manquantes')
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'valeurs_manquantes_matrix.png'), dpi=300, bbox_inches='tight')
            
            plt.show()
        except Exception as e:
            print(f"Impossible de générer la matrice des valeurs manquantes: {e}\n")
    else:
        print("\n• Aucune valeur manquante dans le dataset.")


def analyse_univariee(df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """
    Analyse univariée des variables
    
    Args:
        df: DataFrame pandas à analyser
        output_dir: Répertoire où sauvegarder les figures générées (optionnel)
    """
    print("\n=== ANALYSE UNIVARIÉE ===")
    
    # Variables numériques
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_cols) > 0:
        print("\n• Distribution des variables numériques:")
        
        for i, col in enumerate(num_cols[:10]):  # Limiter à 10 colonnes pour la lisibilité
            plt.figure(figsize=(12, 5))
            
            # Histogramme
            plt.subplot(1, 2, 1)
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Distribution de {col}')
            
            # Boxplot
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col].dropna())
            plt.title(f'Boxplot de {col}')
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'univariee_{col}.png'), dpi=300, bbox_inches='tight')
            
            plt.show()
            
            # Statistiques descriptives
            desc = df[col].describe()
            skewness = stats.skew(df[col].dropna())
            kurtosis = stats.kurtosis(df[col].dropna())
            
            print(f"\n   - {col}:")
            print(f"     Min: {desc['min']:.2f}, Max: {desc['max']:.2f}, Moyenne: {desc['mean']:.2f}, Médiane: {desc['50%']:.2f}")
            print(f"     Écart-type: {desc['std']:.2f}, Asymétrie: {skewness:.2f}, Kurtosis: {kurtosis:.2f}")
            
            # Détection des valeurs aberrantes
            Q1 = desc['25%']
            Q3 = desc['75%']
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            
            if len(outliers) > 0:
                print(f"     Valeurs aberrantes: {len(outliers)} ({len(outliers)/len(df[col].dropna())*100:.2f}%)")
        
        if len(num_cols) > 10:
            print(f"\n   ... {len(num_cols) - 10} autres variables numériques non affichées")
    
    # Variables catégorielles
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        print("\n• Distribution des variables catégorielles:")
        
        for i, col in enumerate(cat_cols[:10]):  # Limiter à 10 colonnes pour la lisibilité
            # Ne pas afficher si trop de valeurs uniques
            if df[col].nunique() <= 20:  # Seuil arbitraire
                plt.figure(figsize=(12, 6))
                countplot = sns.countplot(y=col, data=df, order=df[col].value_counts().index[:20])
                plt.title(f'Distribution de {col}')
                
                # Ajouter les pourcentages
                total = len(df[col].dropna())
                for p in countplot.patches:
                    percentage = f'{100 * p.get_width() / total:.1f}%'
                    x = p.get_width() + total * 0.01
                    y = p.get_y() + p.get_height() / 2
                    countplot.annotate(percentage, (x, y))
                
                plt.tight_layout()
                
                if output_dir:
                    plt.savefig(os.path.join(output_dir, f'univariee_cat_{col}.png'), dpi=300, bbox_inches='tight')
                
                plt.show()
                
                # Afficher les statistiques
                value_counts = df[col].value_counts()
                print(f"\n   - {col}: {df[col].nunique()} valeurs uniques")
                print(f"     Valeur la plus fréquente: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences, {value_counts.iloc[0]/total*100:.2f}%)")
            else:
                print(f"\n   - {col}: {df[col].nunique()} valeurs uniques (trop pour afficher la distribution)")


def analyse_bivariee(df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """
    Analyse bivariée des relations entre variables
    
    Args:
        df: DataFrame pandas à analyser
        output_dir: Répertoire où sauvegarder les figures générées (optionnel)
    """
    print("\n=== ANALYSE BIVARIÉE ===")
    
    # Variables numériques
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(num_cols) > 1:
        print("\n• Relations entre variables numériques:")
        
        # Sélectionner jusqu'à 5 colonnes numériques pour l'analyse
        selected_num_cols = num_cols[:min(5, len(num_cols))]
        
        # Créer un scatter plot matrix
        sns.pairplot(df[selected_num_cols], diag_kind='kde', plot_kws={'alpha': 0.6})
        plt.suptitle('Matrice de scatter plots', y=1.02)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'bivariee_pairplot.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    # Variable catégorielle vs numérique
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(cat_cols) > 0 and len(num_cols) > 0:
        print("\n• Relations entre variables catégorielles et numériques:")
        
        # Sélectionner jusqu'à 3 colonnes numériques et 2 catégorielles
        selected_num_cols = num_cols[:min(3, len(num_cols))]
        selected_cat_cols = [col for col in cat_cols if df[col].nunique() <= 10][:min(2, len(cat_cols))]
        
        for cat_col in selected_cat_cols:
            for num_col in selected_num_cols:
                plt.figure(figsize=(12, 6))
                
                # Boxplot
                plt.subplot(1, 2, 1)
                sns.boxplot(x=cat_col, y=num_col, data=df)
                plt.title(f'Boxplot de {num_col} par {cat_col}')
                plt.xticks(rotation=90)
                
                # Barplot
                plt.subplot(1, 2, 2)
                sns.barplot(x=cat_col, y=num_col, data=df, ci=None)
                plt.title(f'Moyenne de {num_col} par {cat_col}')
                plt.xticks(rotation=90)
                
                plt.tight_layout()
                
                if output_dir:
                    plt.savefig(os.path.join(output_dir, f'bivariee_{cat_col}_{num_col}.png'), dpi=300, bbox_inches='tight')
                
                plt.show()


def analyse_correlations(df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """
    Analyse et visualise les corrélations entre variables numériques
    
    Args:
        df: DataFrame pandas à analyser
        output_dir: Répertoire où sauvegarder les figures générées (optionnel)
    """
    print("\n=== ANALYSE DES CORRÉLATIONS ===")
    
    # Sélectionner les variables numériques
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(num_cols) > 1:
        # Calculer la matrice de corrélation
        corr_matrix = df[num_cols].corr()
        
        # Afficher la matrice de corrélation
        print("\n• Matrice de corrélation:")
        display(corr_matrix.round(2))
        
        # Visualiser la matrice de corrélation avec une heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
                   square=True, linewidths=.5, annot=True, fmt=".2f")
        
        plt.title('Matrice de corrélation')
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Identifier les corrélations les plus fortes
        corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
        # Exclure les auto-corrélations et les doublons
        corr_pairs = corr_pairs[~(corr_pairs == 1.0)]
        # Supprimer les doublons en ne gardant que le triangle supérieur
        corr_pairs = corr_pairs[~mask.reshape(-1)]
        
        if len(corr_pairs) > 0:
            print("\n• Corrélations les plus fortes:")
            for (var1, var2), corr in corr_pairs[:10].items():
                print(f"   - {var1} vs {var2}: {corr:.2f}")
                
                # Visualiser la relation entre les deux variables
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=var1, y=var2, data=df, alpha=0.6)
                plt.title(f'Relation entre {var1} et {var2} (corr = {corr:.2f})')
                plt.grid(True, alpha=0.3)
                
                if output_dir:
                    plt.savefig(os.path.join(output_dir, f'correlation_{var1}_{var2}.png'), dpi=300, bbox_inches='tight')
                
                plt.show()
    else:
        print("\n• Pas assez de variables numériques pour analyser les corrélations.")


def distribution_variables_numeriques(df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """
    Analyse en profondeur la distribution des variables numériques
    
    Args:
        df: DataFrame pandas à analyser
        output_dir: Répertoire où sauvegarder les figures générées (optionnel)
    """
    print("\n=== DISTRIBUTION DES VARIABLES NUMÉRIQUES ===")
    
    # Sélectionner les variables numériques
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(num_cols) > 0:
        for col in num_cols[:5]:  # Limiter à 5 colonnes pour la lisibilité
            print(f"\n• Analyse de la distribution de {col}:")
            
            # Vérifier si la colonne contient suffisamment de valeurs uniques
            if df[col].nunique() <= 1:
                print(f"   Variable constante ({df[col].iloc[0]}), analyse impossible.")
                continue
                
            # Statistiques de base
            stats_desc = df[col].describe()
            print(f"   - Statistiques: Min={stats_desc['min']:.2f}, Max={stats_desc['max']:.2f}, Moyenne={stats_desc['mean']:.2f}, Médiane={stats_desc['50%']:.2f}")
            
            # Mesures de forme
            skewness = stats.skew(df[col].dropna())
            kurtosis = stats.kurtosis(df[col].dropna())
            print(f"   - Forme: Asymétrie={skewness:.2f}, Kurtosis={kurtosis:.2f}")
            
            # Visualisation avancée
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Histogramme
            sns.histplot(df[col].dropna(), kde=True, ax=axes[0, 0])
            axes[0, 0].set_title(f'Distribution de {col}')
            axes[0, 0].axvline(stats_desc['mean'], color='r', linestyle='--', label='Moyenne')
            axes[0, 0].axvline(stats_desc['50%'], color='g', linestyle='-.', label='Médiane')
            axes[0, 0].legend()
            
            # Boxplot
            sns.boxplot(x=df[col].dropna(), ax=axes[0, 1])
            axes[0, 1].set_title(f'Boxplot de {col}')
            
            # QQ-Plot
            stats.probplot(df[col].dropna(), plot=axes[1, 0])
            axes[1, 0].set_title('QQ-Plot (test de normalité)')
            
            # ECDF
            sns.ecdfplot(df[col].dropna(), ax=axes[1, 1])
            axes[1, 1].set_title('Fonction de répartition empirique (ECDF)')
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'distribution_{col}.png'), dpi=300, bbox_inches='tight')
            
            plt.show()
            
            # Test de normalité
            stat, p = stats.shapiro(df[col].dropna().sample(min(5000, len(df[col].dropna()))))
            alpha = 0.05
            if p > alpha:
                print(f"   - Test de Shapiro-Wilk: Distribution normale (p-value = {p:.4f})")
            else:
                print(f"   - Test de Shapiro-Wilk: Distribution non normale (p-value = {p:.4f})")
        
        if len(num_cols) > 5:
            print(f"\n   ... {len(num_cols) - 5} autres variables numériques non affichées")
    else:
        print("\n• Aucune variable numérique dans le dataset.")


def analyse_temporelle(df: pd.DataFrame, col_date: str, output_dir: Optional[str] = None) -> None:
    """
    Analyse temporelle des données
    
    Args:
        df: DataFrame pandas à analyser
        col_date: Nom de la colonne contenant les dates
        output_dir: Répertoire où sauvegarder les figures générées (optionnel)
    """
    print(f"\n=== ANALYSE TEMPORELLE (variable: {col_date}) ===")
    
    # Vérifier que la colonne est bien au format datetime
    if not pd.api.types.is_datetime64_any_dtype(df[col_date]):
        try:
            df[col_date] = pd.to_datetime(df[col_date])
            print(f"Colonne {col_date} convertie en datetime.")
        except:
            print(f"Impossible de convertir la colonne {col_date} en datetime.")
            return
    
    # Créer une copie du dataframe avec la colonne de date comme index
    df_temp = df.copy()
    df_temp.set_index(col_date, inplace=True)
    
    # Sélectionner les variables numériques
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns[:3]  # Limiter à 3 colonnes
    
    if len(num_cols) > 0:
        print("\n• Évolution temporelle des variables numériques:")
        
        for col in num_cols:
            plt.figure(figsize=(14, 6))
            
            # Si le dataset est trop grand, rééchantillonner par mois ou semaine
            if len(df_temp) > 1000:
                # Essayer différentes fréquences de rééchantillonnage selon la plage temporelle
                date_range = df_temp.index.max() - df_temp.index.min()
                
                if date_range.days > 365 * 5:  # Plus de 5 ans
                    resampled = df_temp[col].resample('Q').mean()  # Trimestriel
                    freq_label = "trimestrielle"
                elif date_range.days > 365:  # Plus d'un an
                    resampled = df_temp[col].resample('M').mean()  # Mensuel
                    freq_label = "mensuelle"
                else:
                    resampled = df_temp[col].resample('W').mean()  # Hebdomadaire
                    freq_label = "hebdomadaire"
                
                resampled.plot(marker='o')
                plt.title(f'Évolution {freq_label} de {col}')
            else:
                df_temp[col].plot()
                plt.title(f'Évolution temporelle de {col}')
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'temporel_{col}.png'), dpi=300, bbox_inches='tight')
            
            plt.show()
        
        # Décomposition saisonnière (si applicable)
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            for col in num_cols:
                # Vérifier s'il y a suffisamment de données et pas trop de valeurs manquantes
                if len(df_temp[col].dropna()) > 10:
                    # S'assurer que l'index est régulier
                    if date_range.days > 365:  # Plus d'un an
                        ts = df_temp[col].resample('M').mean()
                        period = 12  # Annuel (12 mois)
                    else:
                        ts = df_temp[col].resample('D').mean()
                        period = 7  # Hebdomadaire
                    
                    # Interpoler les valeurs manquantes
                    ts = ts.interpolate()
                    
                    if len(ts) > period * 2:  # Au moins 2 cycles complets
                        result = seasonal_decompose(ts, model='additive', period=period)
                        
                        plt.figure(figsize=(14, 10))
                        plt.subplot(411)
                        plt.plot(result.observed)
                        plt.title('Série temporelle observée')
                        plt.subplot(412)
                        plt.plot(result.trend)
                        plt.title('Tendance')
                        plt.subplot(413)
                        plt.plot(result.seasonal)
                        plt.title('Saisonnalité')
                        plt.subplot(414)
                        plt.plot(result.resid)
                        plt.title('Résidus')
                        plt.tight_layout()
                        
                        if output_dir:
                            plt.savefig(os.path.join(output_dir, f'decomp_{col}.png'), dpi=300, bbox_inches='tight')
                        
                        plt.show()
        except:
            print("Décomposition saisonnière impossible. Installez statsmodels pour cette fonctionnalité.")
    else:
        print("\n• Aucune variable numérique disponible pour l'analyse temporelle.")


# Si le script est exécuté directement
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyse exploratoire de données")
    parser.add_argument('file', type=str, help='Chemin vers le fichier CSV à analyser')
    parser.add_argument('--titre', type=str, default="Analyse Exploratoire de Données", help='Titre de l\'analyse')
    parser.add_argument('--output', type=str, default=None, help='Répertoire où sauvegarder les figures générées')
    
    args = parser.parse_args()
    
    try:
        df = pd.read_csv(args.file)
        analyse_exploratoire(df, titre=args.titre, output_dir=args.output)
    except Exception as e:
        print(f"Erreur lors de l'analyse: {e}")
