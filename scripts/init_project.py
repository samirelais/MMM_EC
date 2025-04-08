import os
import sys
import shutil
from pyspark.sql import SparkSession
import kaggle
import zipfile
import pandas as pd

# Ajouter le répertoire src au chemin Python
sys.path.append(os.path.abspath("../"))

def setup_project_structure():
    """Configure la structure complète du projet."""
    print("Configuration de la structure du projet...")
    
    # Liste des répertoires à créer
    directories = [
        "data",
        "notebooks",
        "reports",
        "reports/figures",
        "config",
        "src/data",
        "src/features",
        "src/models",
        "src/visualization",
        "src/pipeline",
        "tests",
        "app"
    ]
    
    # Créer les répertoires
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Créé: {directory}/")
    
    print("Structure du projet configurée avec succès !")

def download_online_retail_data():
    """
    Télécharge le jeu de données Online Retail depuis Kaggle et le prépare pour l'analyse.
    """
    print("Téléchargement du jeu de données Online Retail depuis Kaggle...")
    
    # Configurer l'authentification Kaggle
    os.environ['KAGGLE_USERNAME'] = input("Entrez votre nom d'utilisateur Kaggle: ")
    os.environ['KAGGLE_KEY'] = input("Entrez votre clé API Kaggle: ")
    
    # Télécharger les données
    kaggle.api.dataset_download_files('vijayuv/onlineretail', path='data', unzip=True)
    
    print("Données téléchargées avec succès!")
    
    # Charger et convertir les données au format CSV
    print("Conversion des données au format CSV...")
    df = pd.read_excel('data/OnlineRetail.xlsx')
    df.to_csv('data/online_retail.csv', index=False)
    
    print("Données convertie en CSV avec succès!")
    
    return df

def initialize_spark():
    """Initialise et retourne une session Spark."""
    spark = SparkSession.builder \
        .appName("mmm_init") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    return spark

def main():
    """Fonction principale pour initialiser le projet."""
    print("Initialisation du projet Marketing Mix Modeling avec les données Online Retail...")
    
    # Configuration de la structure du projet
    setup_project_structure()
    
    # Téléchargement des données
    try:
        df = download_online_retail_data()
        print(f"Aperçu des données ({df.shape[0]} lignes, {df.shape[1]} colonnes):")
        print(df.head())
    except Exception as e:
        print(f"Erreur lors du téléchargement des données: {e}")
        print("Vous pouvez télécharger manuellement les données depuis https://www.kaggle.com/datasets/vijayuv/onlineretail")
        print("et les placer dans le répertoire 'data/' sous le nom 'online_retail.csv'.")
    
    # Vérifier l'installation des dépendances
    try:
        import lightgbm
        print("LightGBM est correctement installé.")
    except ImportError:
        print("LightGBM n'est pas installé. Veuillez l'installer avec 'pip install lightgbm'.")
    
    # Tester l'initialisation de Spark
    try:
        spark = initialize_spark()
        print(f"Spark initialisé avec succès (version {spark.version}).")
        spark.stop()
    except Exception as e:
        print(f"Erreur lors de l'initialisation de Spark: {e}")
    
    print("\nProjet initialisé avec succès! Pour lancer l'analyse MMM, exécutez:")
    print("python scripts/run_online_retail_mmm.py")

if __name__ == "__main__":
    main()
