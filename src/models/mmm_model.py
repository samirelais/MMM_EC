import os
import json
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.window import Window
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from src.models.adstock import AdstockModels

class MMMModel:
    """
    Classe principale pour le modèle de Marketing Mix Modeling (MMM).
    """
    
    def __init__(self, spark_session, config_path):
        """
        Initialise le modèle avec une session Spark et un chemin de configuration.
        
        Args:
            spark_session: Session Spark active
            config_path: Chemin vers le fichier de configuration JSON
        """
        self.spark = spark_session
        
        # Charger la configuration
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        
        # Initialiser les modèles
        self.model = None
        self.channel_contributions = None
        
        # Stocker les paramètres d'adstock par canal
        self.adstock_params = {
            'tv': {'decay_rate': 0.7, 'max_lag': 14, 'saturation_type': 'hill', 'k': 0.7, 'S': 5000},
            'radio': {'decay_rate': 0.6, 'max_lag': 7, 'saturation_type': 'hill', 'k': 0.6, 'S': 2000},
            'print': {'decay_rate': 0.5, 'max_lag': 21, 'saturation_type': 'hill', 'k': 0.5, 'S': 3000},
            'social_media': {'decay_rate': 0.5, 'max_lag': 5, 'saturation_type': 'hill', 'k': 0.8, 'S': 2500},
            'search': {'decay_rate': 0.4, 'max_lag': 3, 'saturation_type': 'hill', 'k': 0.9, 'S': 4000},
            'email': {'decay_rate': 0.3, 'max_lag': 4, 'saturation_type': 'hill', 'k': 0.7, 'S': 1000},
            'display': {'decay_rate': 0.5, 'max_lag': 10, 'saturation_type': 'hill', 'k': 0.6, 'S': 2000}
        }
    
    def preprocess_data(self, sales_df, marketing_df, external_df=None):
        """
        Prétraite les données pour l'entraînement du modèle.
        
        Args:
            sales_df: DataFrame Spark contenant les données de ventes
            marketing_df: DataFrame Spark contenant les dépenses marketing
            external_df: DataFrame Spark contenant les facteurs externes (optionnel)
            
        Returns:
            DataFrame Spark prétraité
        """
        # Pivoter les données marketing pour avoir une colonne par canal
        pivot_marketing = marketing_df.groupBy("date").pivot("channel").sum("spend")
        
        # Joindre les données de ventes et marketing
        joined_df = sales_df.join(pivot_marketing, on="date", how="inner")
        
        # Joindre les facteurs externes si fournis
        if external_df is not None:
            joined_df = joined_df.join(external_df, on="date", how="left")
        
        # Remplir les valeurs manquantes
        for col_name in joined_df.columns:
            if col_name != "date":
                joined_df = joined_df.withColumn(
                    col_name, 
                    F.coalesce(col_name, F.lit(0))
                )
        
        # Appliquer les transformations adstock et saturation pour chaque canal
        for channel in self.config['marketing_channels']:
            if channel in joined_df.columns:
                params = self.adstock_params.get(channel, {})
                joined_df = AdstockModels.process_channel_spark(joined_df, channel, params)
        
        # Ajouter des caractéristiques temporelles
        joined_df = joined_df.withColumn("day_of_week", F.dayofweek("date"))
        joined_df = joined_df.withColumn("month", F.month("date"))
        joined_df = joined_df.withColumn("is_weekend", F.when(
            (F.col("day_of_week") == 1) | (F.col("day_of_week") == 7), 1).otherwise(0))
        
        # Encodage cyclique des caractéristiques temporelles
        joined_df = joined_df.withColumn(
            "month_sin", F.sin(F.col("month") * 2 * np.pi / 12))
        joined_df = joined_df.withColumn(
            "month_cos", F.cos(F.col("month") * 2 * np.pi / 12))
        joined_df = joined_df.withColumn(
            "day_of_week_sin", F.sin(F.col("day_of_week") * 2 * np.pi / 7))
        joined_df = joined_df.withColumn(
            "day_of_week_cos", F.cos(F.col("day_of_week") * 2 * np.pi / 7))
        
        return joined_df
    
    def prepare_training_data(self, preprocessed_df, target_col="revenue"):
        """
        Prépare les données pour l'entraînement du modèle.
        
        Args:
            preprocessed_df: DataFrame Spark prétraité
            target_col: Nom de la colonne cible
            
        Returns:
            X_train, y_train (DataFrames pandas)
        """
        # Convertir en pandas pour l'entraînement avec LightGBM
        pdf = preprocessed_df.toPandas()
        
        # Séparer les caractéristiques et la cible
        y = pdf[target_col]
        X = pdf.drop([target_col, "date"], axis=1)
        
        return X, y
    
    def train_model(self, train_df, target_col="revenue"):
        """
        Entraîne le modèle MMM sur les données fournies.
        
        Args:
            train_df: DataFrame Spark d'entraînement
            target_col: Nom de la colonne cible
            
        Returns:
            Modèle entraîné et importances des caractéristiques
        """
        # Préparer les données
        X_train, y_train = self.prepare_training_data(train_df, target_col)
        
        # Paramètres LightGBM
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
            'random_state': self.config['modeling'].get('random_state', 42)
        }
        
        # Créer le jeu de données LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        
        # Entraîner le modèle directement sans validation croisée
        num_boost_round = 500  # Nombre d'itérations fixe
        print(f"Entraînement du modèle avec {num_boost_round} itérations...")
        
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round
        )
        
        # Calculer l'importance des caractéristiques
        feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        return self.model, feature_importances
    
    def evaluate_model(self, test_df, target_col="revenue"):
        """
        Évalue le modèle sur les données de test.
        
        Args:
            test_df: DataFrame Spark de test
            target_col: Nom de la colonne cible
            
        Returns:
            Dictionnaire de métriques d'évaluation
        """
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant d'être évalué.")
        
        # Préparer les données de test
        X_test, y_test = self.prepare_training_data(test_df, target_col)
        
        # Faire des prédictions
        y_pred = self.model.predict(X_test)
        
        # Calculer les métriques
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100  # +1 pour éviter division par zéro
        
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        # Créer un DataFrame pour la visualisation
        eval_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred
        })
        
        return metrics
    
    def calculate_channel_contributions(self, model, full_df):
        """
        Calcule les contributions de chaque canal marketing aux ventes.
        
        Args:
            model: Modèle entraîné
            full_df: DataFrame Spark complet
            
        Returns:
            DataFrame pandas des contributions par canal
        """
        # Convertir en pandas
        pdf = full_df.toPandas()
        
        # Extraire les colonnes de date et de revenu
        date_series = pdf['date']
        actual_revenue = pdf['revenue']
        
        # Préparer les données pour la prédiction
        X = pdf.drop(['date', 'revenue'], axis=1)
        
        # Prédire le revenu total
        y_pred = model.predict(X)
        
        # Calculer la baseline (sans effet marketing)
        X_baseline = X.copy()
        
        # Contribution par canal
        contributions = {}
        contributions['date'] = date_series
        contributions['actual_revenue'] = actual_revenue
        contributions['predicted_revenue'] = y_pred
        
        # Pour chaque canal marketing, calculer sa contribution
        for channel in self.config['marketing_channels']:
            adstock_col = f"{channel}_adstock"
            
            if adstock_col in X.columns:
                # Créer une version des données sans ce canal
                X_without_channel = X.copy()
                X_without_channel[adstock_col] = 0
                
                # Prédire sans ce canal
                y_pred_without_channel = model.predict(X_without_channel)
                
                # La contribution est la différence
                channel_contribution = y_pred - y_pred_without_channel
                
                # Stocker la contribution
                contributions[f"{channel}_contribution"] = channel_contribution
                
                # Calculer le ROI
                spend_col = channel
                if spend_col in X.columns:
                    spend = X[spend_col]
                    # Éviter division par zéro
                    masked_spend = spend.copy()
                    masked_spend[masked_spend == 0] = np.nan
                    roi = channel_contribution / masked_spend
                    contributions[f"{channel}_roi"] = roi
        
        # Calculer la contribution de base (non attribuée aux canaux marketing)
        total_marketing_contribution = sum([contributions[f"{channel}_contribution"] 
                                           for channel in self.config['marketing_channels'] 
                                           if f"{channel}_contribution" in contributions])
        
        contributions['baseline_contribution'] = y_pred - total_marketing_contribution
        
        # Convertir en DataFrame
        contributions_df = pd.DataFrame(contributions)
        
        # Calcul des pourcentages de contribution
        contribution_cols = [f"{channel}_contribution" for channel in self.config['marketing_channels'] 
                            if f"{channel}_contribution" in contributions]
        contribution_cols.append('baseline_contribution')
        
        for col in contribution_cols:
            contributions_df[col + '_pct'] = contributions_df[col] / contributions_df['predicted_revenue'] * 100
        
        self.channel_contributions = contributions_df
        return contributions_df
    
    def optimize_budget(self, contributions_df, total_budget=None):
        """
        Optimise l'allocation budgétaire en fonction des contributions des canaux.
        
        Args:
            contributions_df: DataFrame pandas des contributions
            total_budget: Budget total à allouer (par défaut, utilise le budget de la configuration)
            
        Returns:
            DataFrame pandas avec l'allocation budgétaire optimisée
        """
        if total_budget is None:
            total_budget = self.config['optimization'].get('budget_constraint', 100000)
        
        # Extraire les ROI moyens par canal
        roi_data = {}
        for channel in self.config['marketing_channels']:
            roi_col = f"{channel}_roi"
            if roi_col in contributions_df.columns:
                # Utiliser la médiane pour être robuste aux valeurs aberrantes
                roi = contributions_df[roi_col].median()
                roi_data[channel] = roi
        
        # Trier les canaux par ROI
        sorted_channels = sorted(roi_data.items(), key=lambda x: x[1], reverse=True)
        
        # Contraintes de budget par canal
        min_pct = self.config['optimization'].get('min_channel_budget_pct', 0.05)
        max_pct = self.config['optimization'].get('max_channel_budget_pct', 0.4)
        
        min_budget = total_budget * min_pct
        max_budget = total_budget * max_pct
        
        # Allocation basée sur le ROI
        # D'abord, allouer le budget minimum à tous les canaux
        allocation = {channel: min_budget for channel, _ in sorted_channels}
        
        # Budget restant après allocation minimale
        remaining_budget = total_budget - (min_budget * len(sorted_channels))
        
        # Allouer le reste proportionnellement au ROI
        if remaining_budget > 0:
            total_roi = sum(roi for _, roi in sorted_channels if roi > 0)
            
            if total_roi > 0:
                for channel, roi in sorted_channels:
                    if roi > 0:
                        # Allocation proportionnelle au ROI
                        additional_budget = remaining_budget * (roi / total_roi)
                        allocation[channel] += additional_budget
                        
                        # Limiter au budget maximum par canal
                        if allocation[channel] > max_budget:
                            overflow = allocation[channel] - max_budget
                            allocation[channel] = max_budget
                            remaining_budget = overflow
                        else:
                            remaining_budget -= additional_budget
            
            # S'il reste du budget, l'allouer équitablement
            if remaining_budget > 0 and len(sorted_channels) > 0:
                per_channel_remaining = remaining_budget / len(sorted_channels)
                for channel in allocation:
                    if allocation[channel] < max_budget:
                        additional = min(per_channel_remaining, max_budget - allocation[channel])
                        allocation[channel] += additional
        
        # Créer un DataFrame pour le résultat
        result = []
        for channel, budget in allocation.items():
            # Calculer le pourcentage du budget total
            budget_pct = budget / total_budget * 100
            
            # Ajouter les données ROI
            roi = roi_data.get(channel, 0)
            
            result.append({
                'channel': channel,
                'budget': budget,
                'budget_pct': budget_pct,
                'roi': roi
            })
        
        result_df = pd.DataFrame(result)
        return result_df.sort_values('budget', ascending=False)