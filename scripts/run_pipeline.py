from pyspark.sql import SparkSession
import os
import sys
import json

# Ajouter le répertoire src au chemin Python
sys.path.append(os.path.abspath("../"))

from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.features.feature_engineer import FeatureEngineer
# Importer les autres modules nécessaires au fur et à mesure

def load_config(config_path):
    """Charge le fichier de configuration JSON."""
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def initialize_spark(config):
    """Initialise et retourne une session Spark."""
    spark_config = config.get("spark", {})
    
    spark = SparkSession.builder \
        .appName(spark_config.get("app_name", "mmm_ecommerce")) \
        .config("spark.driver.memory", spark_config.get("driver_memory", "4g")) \
        .config("spark.executor.memory", spark_config.get("executor_memory", "4g")) \
        .config("spark.executor.cores", spark_config.get("executor_cores", 2)) \
        .getOrCreate()
    
    return spark

def main():
    """Fonction principale qui exécute le pipeline complet."""
    print("Démarrage du pipeline MMM...")
    
    # Charger la configuration
    config_path = "../config/mmm_config.json"
    config = load_config(config_path)
    
    # Initialiser Spark
    spark = initialize_spark(config)
    print(f"Session Spark initialisée: {spark.version}")
    
    # Créer les instances des différentes classes
    data_loader = DataLoader(spark, config_path)
    data_cleaner = DataCleaner(config_path)
    feature_engineer = FeatureEngineer(config_path)
    
    # Charger les données
    print("Chargement des données...")
    sales_df = data_loader.load_sales_data()
    marketing_df = data_loader.load_marketing_data()
    external_df = data_loader.load_external_factors()
    
    print(f"Données chargées: {sales_df.count()} lignes de ventes, {marketing_df.count()} lignes marketing")
    
    # Nettoyer les données
    print("Nettoyage des données...")
    sales_df = data_cleaner.remove_outliers(sales_df, "revenue")
    sales_df = data_cleaner.fill_missing_values(sales_df)
    
    marketing_df = data_cleaner.fill_missing_values(marketing_df)
    
    # Feature engineering
    print("Création des caractéristiques...")
    marketing_df = feature_engineer.create_lag_features(
        marketing_df, 
        id_cols=["channel"], 
        target_cols=["spend"]
    )
    
    marketing_df = feature_engineer.create_rolling_features(
        marketing_df,
        id_cols=["channel"],
        target_cols=["spend"]
    )
    
    # Ajouter des caractéristiques temporelles
    sales_df = feature_engineer.create_seasonality_features(sales_df)
    sales_df = feature_engineer.create_holiday_features(sales_df)
    
    # Joindre les données
    print("Fusion des données...")
    # Code pour joindre les dataframes
    
    # Sauvegarder les données préparées
    print("Sauvegarde des données préparées...")
    # Code pour sauvegarder les dataframes
    
    print("Pipeline MMM terminé avec succès!")
    
    # Arrêter la session Spark
    spark.stop()

if __name__ == "__main__":
    main()
