"""
Fonctions pour charger différents datasets publics ou personnels.
"""

import os
import pandas as pd
from typing import Optional, Union, Dict, List, Tuple
import urllib.request
import zipfile
import io


def load_dataset(path: str, **kwargs) -> pd.DataFrame:
    """
    Charge un dataset à partir d'un chemin local ou d'une URL.
    
    Args:
        path: Chemin local ou URL vers le dataset
        **kwargs: Arguments supplémentaires pour pd.read_csv ou pd.read_excel
    
    Returns:
        DataFrame pandas contenant les données
    """
    # Détecter si c'est une URL ou un chemin local
    is_url = path.startswith(('http://', 'https://'))
    
    # Déterminer l'extension du fichier
    file_ext = os.path.splitext(path)[1].lower()
    
    if is_url:
        if file_ext in ['.csv', '.txt']:
            return pd.read_csv(path, **kwargs)
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(path, **kwargs)
        elif file_ext == '.json':
            return pd.read_json(path, **kwargs)
        elif file_ext == '.parquet':
            return pd.read_parquet(path, **kwargs)
        else:
            raise ValueError(f"Format de fichier non supporté: {file_ext}")
    else:
        # Chemin local
        if not os.path.exists(path):
            raise FileNotFoundError(f"Le fichier {path} n'existe pas")
        
        if file_ext in ['.csv', '.txt']:
            return pd.read_csv(path, **kwargs)
        elif file_ext in ['.xlsx', '.xls']:
            return pd.read_excel(path, **kwargs)
        elif file_ext == '.json':
            return pd.read_json(path, **kwargs)
        elif file_ext == '.parquet':
            return pd.read_parquet(path, **kwargs)
        else:
            raise ValueError(f"Format de fichier non supporté: {file_ext}")


def load_spotify() -> pd.DataFrame:
    """
    Charge le dataset Spotify avec des informations sur les artistes et les chansons.
    
    Cette fonction recherche le fichier spotifydataset.csv à la racine du projet et le charge
    en tant que DataFrame pandas. Le dataset contient des informations sur les artistes, chansons,
    et caractéristiques audio comme la danceability, energy, tempo, etc.
    
    Returns:
        DataFrame pandas contenant les données Spotify avec les colonnes suivantes:
        - artist_name: Nom de l'artiste
        - genres: Liste des genres associés à l'artiste (format texte)
        - followers: Nombre d'abonnés de l'artiste
        - artist_popularity: Score de popularité de l'artiste (0-100)
        - track_name: Nom de la chanson
        - album_name: Nom de l'album
        - release_date: Date de sortie de l'album
        - plus diverses caractéristiques audio (danceability, energy, etc.)
    
    Raises:
        FileNotFoundError: Si le fichier spotifydataset.csv n'est pas trouvé
    
    Example:
        >>> df = load_spotify()
        >>> print(f"Nombre d'artistes: {df['artist_name'].nunique()}")
        >>> print(f"Caractéristiques disponibles: {df.columns.tolist()}")
    """
    # Utiliser le chemin relatif au projet
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    file_path = os.path.join(root_dir, 'spotifydataset.csv')
    
    # Vérifier si le fichier existe
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} n'existe pas")
    
    # Charger le dataset
    return pd.read_csv(file_path)


def load_iris() -> pd.DataFrame:
    """
    Charge le dataset Iris classique.
    
    Returns:
        DataFrame pandas contenant les données Iris
    """
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    return pd.read_csv(url)


def load_titanic() -> pd.DataFrame:
    """
    Charge le dataset Titanic classique.
    
    Returns:
        DataFrame pandas contenant les données Titanic
    """
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)


def load_tips() -> pd.DataFrame:
    """
    Charge le dataset Tips de Seaborn.
    
    Returns:
        DataFrame pandas contenant les données Tips
    """
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv"
    return pd.read_csv(url)


def load_boston_housing() -> pd.DataFrame:
    """
    Charge le dataset Boston Housing.
    
    Returns:
        DataFrame pandas contenant les données Boston Housing
    """
    url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
    return pd.read_csv(url)


def load_from_kaggle(dataset_name: str, 
                    api_key_path: Optional[str] = None,
                    file_name: Optional[str] = None) -> pd.DataFrame:
    """
    Charge un dataset depuis Kaggle.
    Nécessite l'API Kaggle et les identifiants.
    
    Args:
        dataset_name: Nom du dataset sur Kaggle (ex: "username/dataset-name")
        api_key_path: Chemin vers le fichier kaggle.json (par défaut: ~/.kaggle/kaggle.json)
        file_name: Nom du fichier à télécharger (si le dataset contient plusieurs fichiers)
    
    Returns:
        DataFrame pandas contenant les données
    """
    try:
        import kaggle
    except ImportError:
        raise ImportError("Le package 'kaggle' est requis. Installez-le avec 'pip install kaggle'")
    
    # Configurer l'API Kaggle
    if api_key_path:
        os.environ['KAGGLE_CONFIG_DIR'] = os.path.dirname(api_key_path)
    
    # Créer le dossier de destination s'il n'existe pas
    os.makedirs("data/raw", exist_ok=True)
    
    # Télécharger le dataset
    kaggle.api.dataset_download_files(dataset_name, path="data/raw", unzip=True)
    
    # Si le nom de fichier est spécifié, charger ce fichier
    if file_name:
        file_path = os.path.join("data/raw", file_name)
    else:
        # Sinon, essayer de trouver un fichier CSV ou Excel
        files = os.listdir("data/raw")
        csv_files = [f for f in files if f.endswith('.csv')]
        excel_files = [f for f in files if f.endswith(('.xlsx', '.xls'))]
        
        if csv_files:
            file_path = os.path.join("data/raw", csv_files[0])
        elif excel_files:
            file_path = os.path.join("data/raw", excel_files[0])
        else:
            raise ValueError("Aucun fichier CSV ou Excel trouvé dans le dataset")
    
    # Charger le fichier
    return load_dataset(file_path)


def download_and_load(url: str, 
                     file_name: Optional[str] = None,
                     extract_zip: bool = False,
                     **kwargs) -> pd.DataFrame:
    """
    Télécharge un fichier depuis une URL et le charge en mémoire.
    
    Args:
        url: URL du fichier à télécharger
        file_name: Nom de fichier à utiliser pour l'enregistrement local (facultatif)
        extract_zip: Si True, extrait le contenu d'un fichier ZIP
        **kwargs: Arguments supplémentaires pour pd.read_csv ou pd.read_excel
    
    Returns:
        DataFrame pandas contenant les données
    """
    # Créer le dossier de destination s'il n'existe pas
    os.makedirs("data/raw", exist_ok=True)
    
    # Déterminer le nom de fichier
    if not file_name:
        file_name = url.split('/')[-1]
    
    file_path = os.path.join("data/raw", file_name)
    
    # Télécharger le fichier
    print(f"Téléchargement depuis {url}...")
    urllib.request.urlretrieve(url, file_path)
    print(f"Fichier téléchargé dans {file_path}")
    
    # Si c'est un fichier ZIP, l'extraire
    if extract_zip or file_name.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            extract_dir = os.path.join("data/raw", os.path.splitext(file_name)[0])
            os.makedirs(extract_dir, exist_ok=True)
            zip_ref.extractall(extract_dir)
            print(f"Fichier ZIP extrait dans {extract_dir}")
            
            # Essayer de trouver un fichier CSV ou Excel dans le dossier extrait
            files = os.listdir(extract_dir)
            csv_files = [f for f in files if f.endswith('.csv')]
            excel_files = [f for f in files if f.endswith(('.xlsx', '.xls'))]
            
            if csv_files:
                file_path = os.path.join(extract_dir, csv_files[0])
            elif excel_files:
                file_path = os.path.join(extract_dir, excel_files[0])
            else:
                raise ValueError("Aucun fichier CSV ou Excel trouvé dans l'archive ZIP")
    
    return load_dataset(file_path, **kwargs)


def save_dataset(df: pd.DataFrame, 
                path: str, 
                index: bool = False, 
                **kwargs) -> None:
    """
    Sauvegarde un DataFrame dans un fichier CSV, Excel, ou autre format.
    
    Args:
        df: DataFrame pandas à sauvegarder
        path: Chemin où sauvegarder le fichier
        index: Si True, inclut l'index dans le fichier sauvegardé
        **kwargs: Arguments supplémentaires pour la méthode de sauvegarde
    """
    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Déterminer l'extension du fichier
    file_ext = os.path.splitext(path)[1].lower()
    
    if file_ext in ['.csv', '.txt']:
        df.to_csv(path, index=index, **kwargs)
    elif file_ext in ['.xlsx', '.xls']:
        df.to_excel(path, index=index, **kwargs)
    elif file_ext == '.json':
        df.to_json(path, **kwargs)
    elif file_ext == '.parquet':
        df.to_parquet(path, **kwargs)
    else:
        raise ValueError(f"Format de fichier non supporté: {file_ext}")
    
    print(f"Dataset sauvegardé dans {path}")