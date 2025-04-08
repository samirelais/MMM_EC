import os
import kaggle
import zipfile
import pandas as pd

def download_online_retail_data():
    """
    Télécharge le jeu de données Online Retail depuis Kaggle et le prépare pour l'analyse.
    """
    print("Téléchargement du jeu de données Online Retail depuis Kaggle...")
    
    # Créer le dossier data s'il n'existe pas
    os.makedirs('data', exist_ok=True)
    
    # Télécharger les données
    kaggle.api.dataset_download_files('vijayuv/onlineretail', path='data', unzip=True)
    
    # Charger les données
    print("Chargement des données...")
    df = pd.read_excel('data/OnlineRetail.xlsx')
    
    # Afficher un aperçu des données
    print("Aperçu des données :")
    print(df.head())
    print("\nInformations sur les données :")
    print(df.info())
    print("\nStatistiques descriptives :")
    print(df.describe())
    
    # Sauvegarder au format CSV pour une utilisation plus facile avec PySpark
    print("Sauvegarde des données au format CSV...")
    df.to_csv('data/online_retail.csv', index=False)
    
    print("Téléchargement et préparation des données terminés.")
    return df

if __name__ == "__main__":
    download_online_retail_data()
