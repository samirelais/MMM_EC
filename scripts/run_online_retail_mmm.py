from pyspark.sql import SparkSession
import os
import sys
import json

# Ajouter le répertoire src au chemin Python
sys.path.append(os.path.abspath("../"))

from src.data.online_retail_loader import OnlineRetailLoader
from src.data.data_cleaner import DataCleaner
from src.features.feature_engineer import FeatureEngineer
from src.models.mmm_model import MMMModel
from src.visualization.visualization import MMMVisualization

def load_config(config_path):
   """Charge le fichier de configuration JSON."""
   with open(config_path, 'r') as config_file:
       return json.load(config_file)

def initialize_spark(config):
   """Initialise et retourne une session Spark."""
   spark_config = config.get("spark", {})
   
   spark = SparkSession.builder \
       .appName(spark_config.get("app_name", "mmm_online_retail")) \
       .config("spark.driver.memory", spark_config.get("driver_memory", "4g")) \
       .config("spark.executor.memory", spark_config.get("executor_memory", "4g")) \
       .config("spark.executor.cores", spark_config.get("executor_cores", 2)) \
       .getOrCreate()
   
   return spark

def main():
   """Fonction principale qui exécute le pipeline MMM complet pour les données Online Retail."""
   print("Démarrage du pipeline MMM pour les données Online Retail...")
   
   # Charger la configuration
   config_path = "../config/online_retail_config.json"
   config = load_config(config_path)
   
   # Initialiser Spark
   spark = initialize_spark(config)
   print(f"Session Spark initialisée: {spark.version}")
   
   # Créer les instances des différentes classes
   retail_loader = OnlineRetailLoader(spark, config_path)
   data_cleaner = DataCleaner(config_path)
   feature_engineer = FeatureEngineer(config_path)
   mmm_model = MMMModel(spark, config_path)
   visualizer = MMMVisualization(config_path)
   
   # Charger les données retail
   print("Chargement des données Online Retail...")
   retail_df = retail_loader.load_retail_data()
   
   # Créer les données de ventes quotidiennes
   print("Création des données de ventes quotidiennes...")
   sales_df = retail_loader.create_daily_sales_data(retail_df)
   
   # Simuler les données marketing basées sur les patterns de vente
   print("Simulation des données marketing...")
   marketing_df = retail_loader.create_marketing_channel_data(retail_df)
   
   # Simuler les facteurs externes
   print("Simulation des facteurs externes...")
   external_df = retail_loader.create_external_factors(retail_df)
   
   # Nettoyer les données
   print("Nettoyage des données...")
   sales_df = data_cleaner.remove_outliers(sales_df, "revenue")
   sales_df = data_cleaner.fill_missing_values(sales_df)
   
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
   
   # Prétraiter les données pour le modèle
   print("Prétraitement des données pour le modèle...")
   preprocessed_df = mmm_model.preprocess_data(sales_df, marketing_df, external_df)
   
   # Diviser en ensembles d'entraînement et de test
   train_end_date = config['data']['train_end_date']
   train_df = preprocessed_df.filter(f"date <= '{train_end_date}'")
   test_df = preprocessed_df.filter(f"date > '{train_end_date}'")
   
   print(f"Ensemble d'entraînement: {train_df.count()} lignes")
   print(f"Ensemble de test: {test_df.count()} lignes")
   
   # Entraîner le modèle
   print("Entraînement du modèle MMM...")
   model, feature_importances = mmm_model.train_model(train_df, target_col="revenue")
   
   # Évaluer le modèle
   print("Évaluation du modèle...")
   metrics = mmm_model.evaluate_model(model, test_df, target_col="revenue")
   print(f"Métriques d'évaluation: {metrics}")
   
   # Calculer les contributions des canaux
   print("Calcul des contributions des canaux...")
   contributions = mmm_model.calculate_channel_contributions(model, preprocessed_df)
   
   # Optimiser l'allocation budgétaire
   print("Optimisation de l'allocation budgétaire...")
   budget_allocation = mmm_model.optimize_budget(contributions)
   
   # Générer les visualisations
   print("Génération des visualisations...")
   visualizer.plot_channel_contributions(contributions)
   visualizer.plot_roi_by_channel(contributions)
   visualizer.plot_budget_allocation(budget_allocation)
   visualizer.plot_actual_vs_predicted(preprocessed_df, model)
   
   # Générer un rapport
   print("Génération du rapport...")
   report_path = visualizer.generate_report(
       metrics, 
       contributions, 
       feature_importances, 
       budget_allocation
   )
   
   print(f"Pipeline MMM terminé avec succès! Rapport disponible à {report_path}")
   
   # Arrêter la session Spark
   spark.stop()

if __name__ == "__main__":
   main()
