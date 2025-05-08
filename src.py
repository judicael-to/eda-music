import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno
import warnings
from IPython.display import display, HTML

# Ignorer les avertissements pour une sortie plus propre
warnings.filterwarnings('ignore')

# Configurer le style de visualisation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Fonction principale pour l'analyse exploratoire
def analyse_exploratoire(df, titre="Analyse Exploratoire de Données"):
    """
    Réalise une analyse exploratoire complète sur un dataframe
    
    Args:
        df: DataFrame pandas à analyser
        titre: Titre de l'analyse
    """
    print(f"==== {titre} ====\n")
    
    # 1. Aperçu des données
    apercu_donnees(df)
    
    # 2. Analyse des valeurs manquantes
    analyse_valeurs_manquantes(df)
    
    # 3. Analyse univariée
    analyse_univariee(df)
    
    # 4. Analyse bivariée
    analyse_bivariee(df)
    
    # 5. Analyse des corrélations
    analyse_correlations(df)
    
    # 6. Distribution des variables numériques
    distribution_variables_numeriques(df)
    
    # 7. Analyse temporelle (si applicable)
    colonnes_date = [col for col in df.columns if df[col].dtype == 'datetime64[ns]']
    if colonnes_date:
        for col in colonnes_date:
            analyse_temporelle(df, col)
    
    print("\n==== Fin de l'analyse exploratoire ====")

def apercu_donnees(df):
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

def analyse_valeurs_manquantes(df):
    """Analyse et visualise les valeurs manquantes"""
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
        plt.show()
        
        # Visualisation de la matrice des valeurs manquantes
        try:
            plt.figure(figsize=(12, 8))
            msno.matrix(df)
            plt.title('Matrice des valeurs manquantes')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Impossible de générer la matrice des valeurs manquantes: {e}\n")
    else:
        print("\n• Aucune valeur manquante dans le dataset.")

def analyse_univariee(df):
    """Analyse univariée des variables"""
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
                plt.show()
                
                # Afficher les statistiques
                value_counts = df[col].value_counts()
                print(f"\n   - {col}: {df[col].nunique()} valeurs uniques")
                print(f"     Valeur la plus fréquente: {value_counts.index[0]} ({value_counts.iloc[0]} occurrences, {value_counts.iloc[0]/total*100:.2f}%)")
            else:
                print(f"\n   - {col}: {df[col].nunique()} valeurs uniques (trop nombreuses pour affichage)")
        
        if len(cat_cols) > 10:
            print(f"\n   ... {len(cat_cols) - 10} autres variables catégorielles non affichées")

def analyse_bivariee(df):
    """Analyse des relations entre variables"""
    print("\n=== ANALYSE BIVARIÉE ===")
    
    # Variables numériques
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(num_cols) >= 2:
        # Sélectionner maximum 6 variables pour la lisibilité
        selected_num_cols = num_cols[:min(6, len(num_cols))]
        
        try:
            # Pairplot pour relations entre variables numériques
            plt.figure(figsize=(15, 15))
            sns.pairplot(df[selected_num_cols], diag_kind='kde', plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k'})
            plt.suptitle('Relations entre variables numériques', y=1.02)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Impossible de générer le pairplot: {e}")
    
    # Relations entre variables catégorielles et numériques
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(cat_cols) > 0 and len(num_cols) > 0:
        print("\n• Relations entre variables catégorielles et numériques:")
        
        # Sélectionner les variables catégorielles avec peu de valeurs uniques
        filtered_cat_cols = [col for col in cat_cols if df[col].nunique() <= 10][:3]
        
        for cat_col in filtered_cat_cols:
            for num_col in num_cols[:3]:  # Limiter à 3 variables numériques
                plt.figure(figsize=(12, 6))
                
                try:
                    # Boxplot
                    plt.subplot(1, 2, 1)
                    sns.boxplot(x=cat_col, y=num_col, data=df)
                    plt.title(f'{num_col} par {cat_col}')
                    plt.xticks(rotation=45)
                    
                    # Violinplot
                    plt.subplot(1, 2, 2)
                    sns.violinplot(x=cat_col, y=num_col, data=df, inner="quartile")
                    plt.title(f'Distribution de {num_col} par {cat_col}')
                    plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # ANOVA pour tester la significativité des différences
                    categories = df[cat_col].dropna().unique()
                    if len(categories) >= 2:  # Au moins 2 catégories pour comparer
                        try:
                            groups = [df[df[cat_col] == cat][num_col].dropna() for cat in categories]
                            f_val, p_val = stats.f_oneway(*[g for g in groups if len(g) > 0])
                            print(f"   - {num_col} vs {cat_col}: ANOVA F={f_val:.2f}, p-value={p_val:.4f}")
                            if p_val < 0.05:
                                print(f"     Différence significative entre les groupes (p<0.05)")
                            else:
                                print(f"     Pas de différence significative entre les groupes (p≥0.05)")
                        except:
                            print(f"   - {num_col} vs {cat_col}: Impossible de réaliser l'ANOVA")
                except Exception as e:
                    print(f"   - {num_col} vs {cat_col}: Erreur lors de la visualisation: {e}")

def analyse_correlations(df):
    """Analyse des corrélations entre variables numériques"""
    print("\n=== ANALYSE DES CORRÉLATIONS ===")
    
    # Sélectionner uniquement les variables numériques
    num_df = df.select_dtypes(include=['int64', 'float64'])
    
    if num_df.shape[1] >= 2:
        # Calculer la matrice de corrélation
        corr_matrix = num_df.corr()
        
        # Visualiser la matrice de corrélation
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    annot=True, fmt=".2f", square=True, linewidths=.5)
        
        plt.title('Matrice de corrélation entre variables numériques')
        plt.tight_layout()
        plt.show()
        
        # Identifier les paires de variables fortement corrélées
        print("\n• Corrélations significatives (|r| > 0.5):")
        strong_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.5:
                    strong_corr.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if strong_corr:
            for var1, var2, corr in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
                print(f"   - {var1} et {var2}: r = {corr:.3f}")
                
                # Visualiser la relation
                plt.figure(figsize=(10, 6))
                sns.regplot(x=var1, y=var2, data=df, line_kws={"color":"red"})
                plt.title(f'Relation entre {var1} et {var2} (r = {corr:.3f})')
                plt.tight_layout()
                plt.show()
        else:
            print("   Aucune corrélation forte détectée (|r| > 0.5)")
    else:
        print("   Pas assez de variables numériques pour calculer des corrélations.")

def distribution_variables_numeriques(df):
    """Analyse plus détaillée de la distribution des variables numériques"""
    print("\n=== DISTRIBUTION DES VARIABLES NUMÉRIQUES ===")
    
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    if len(num_cols) > 0:
        # QQ plots pour vérifier la normalité
        for col in num_cols[:6]:  # Limiter à 6 variables
            plt.figure(figsize=(12, 5))
            
            # QQ Plot
            plt.subplot(1, 2, 1)
            stats.probplot(df[col].dropna(), plot=plt)
            plt.title(f'QQ Plot de {col}')
            
            # Distribution avec courbe normale superposée
            plt.subplot(1, 2, 2)
            sns.histplot(df[col].dropna(), kde=True)
            
            # Superposer une courbe normale
            x = np.linspace(df[col].min(), df[col].max(), 100)
            mean = df[col].mean()
            std = df[col].std()
            p = stats.norm.pdf(x, mean, std)
            p = p * (df[col].count() * (df[col].max() - df[col].min()) / 10)  # Mise à l'échelle
            plt.plot(x, p, 'r-', linewidth=2)
            
            plt.title(f'Distribution de {col} vs Normale')
            plt.tight_layout()
            plt.show()
            
            # Test de normalité de Shapiro-Wilk (si moins de 5000 observations)
            sample = df[col].dropna()
            if len(sample) > 5000:
                sample = sample.sample(5000)  # Échantillonnage pour le test
                
            try:
                stat, p = stats.shapiro(sample)
                print(f"   - {col}: Test de Shapiro-Wilk - Statistique={stat:.4f}, p-value={p:.4e}")
                if p < 0.05:
                    print(f"     Distribution non normale (p<0.05)")
                else:
                    print(f"     Distribution peut être considérée comme normale (p≥0.05)")
            except Exception as e:
                print(f"   - {col}: Impossible de réaliser le test de normalité: {e}")

def analyse_temporelle(df, col_date):
    """Analyse des séries temporelles (si applicable)"""
    print(f"\n=== ANALYSE TEMPORELLE ({col_date}) ===")
    
    try:
        # S'assurer que la colonne est bien de type datetime
        if df[col_date].dtype != 'datetime64[ns]':
            df[col_date] = pd.to_datetime(df[col_date], errors='coerce')
        
        # Créer des colonnes temporelles
        df_time = df.copy()
        df_time['year'] = df_time[col_date].dt.year
        df_time['month'] = df_time[col_date].dt.month
        df_time['dayofweek'] = df_time[col_date].dt.dayofweek
        df_time['quarter'] = df_time[col_date].dt.quarter
        
        # Sélectionner une variable numérique pour l'analyse temporelle
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(num_cols) > 0:
            target_col = num_cols[0]  # Prendre la première variable numérique
            
            # Distribution par année
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='year', y=target_col, data=df_time)
            plt.title(f'{target_col} par année')
            plt.tight_layout()
            plt.show()
            
            # Distribution par mois
            plt.figure(figsize=(14, 6))
            sns.boxplot(x='month', y=target_col, data=df_time)
            plt.title(f'{target_col} par mois')
            plt.xticks(range(12), ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 'Juil', 'Août', 'Sept', 'Oct', 'Nov', 'Déc'])
            plt.tight_layout()
            plt.show()
            
            # Distribution par jour de la semaine
            plt.figure(figsize=(12, 6))
            sns.boxplot(x='dayofweek', y=target_col, data=df_time)
            plt.title(f'{target_col} par jour de la semaine')
            plt.xticks(range(7), ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim'])
            plt.tight_layout()
            plt.show()
            
            # Evolution temporelle (si assez de données)
            if df_time['year'].nunique() > 1 or df_time['month'].nunique() > 1:
                plt.figure(figsize=(14, 6))
                
                # Agrégation par mois ou par jour selon la granularité des données
                if df_time['year'].nunique() > 1:
                    time_series = df_time.groupby(df_time[col_date].dt.to_period('M')).agg({target_col: 'mean'})
                    time_series.index = time_series.index.to_timestamp()
                else:
                    time_series = df_time.groupby(df_time[col_date].dt.to_period('D')).agg({target_col: 'mean'})
                    time_series.index = time_series.index.to_timestamp()
                
                plt.plot(time_series.index, time_series[target_col], marker='o', linestyle='-')
                plt.title(f'Évolution temporelle de {target_col}')
                plt.grid(True)
                plt.tight_layout()
                plt.show()
    except Exception as e:
        print(f"Erreur lors de l'analyse temporelle: {e}")

# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacer par le chemin vers votre dataset
    # Exemples de datasets publics:
    # - "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    # - "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    # - "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    
    dataset_path = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    
    # Charger les données
    try:
        df = pd.read_csv(dataset_path)
        
        # Exécuter l'analyse exploratoire
        analyse_exploratoire(df, titre=f"Analyse du dataset: {dataset_path.split('/')[-1]}")
    except Exception as e:
        print(f"Erreur lors du chargement ou de l'analyse du dataset: {e}")