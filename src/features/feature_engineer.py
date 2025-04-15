from pyspark.sql import functions as F
from pyspark.sql.window import Window
import json

class FeatureEngineer:
    """
    Classe pour créer des caractéristiques avancées pour le MMM.
    """
    
    def __init__(self, config_path):
        """
        Initialise le FeatureEngineer avec un chemin de configuration.
        
        Args:
            config_path: Chemin vers le fichier de configuration JSON
        """
        # Charger la configuration
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
    
    def create_lag_features(self, df, id_cols, target_cols, lag_periods=None):
        """
        Crée des caractéristiques décalées (lag features) pour les colonnes cibles.
        
        Args:
            df: DataFrame Spark avec les données temporelles
            id_cols: Liste des colonnes d'identification (e.g., 'channel')
            target_cols: Liste des colonnes pour lesquelles créer des lags
            lag_periods: Liste des périodes de décalage en jours
            
        Returns:
            DataFrame Spark avec les caractéristiques décalées
        """
        if lag_periods is None:
            lag_periods = self.config['feature_engineering'].get('lag_periods', [1, 7, 14, 28])
        
        result_df = df
        
        # Pour chaque colonne cible
        for col in target_cols:
            # Définir une fenêtre partitionnée par les colonnes d'ID et ordonnée par date
            window_spec = Window.partitionBy(id_cols).orderBy('date')
            
            # Pour chaque période de décalage
            for lag in lag_periods:
                lag_col_name = f"{col}_lag_{lag}"
                result_df = result_df.withColumn(lag_col_name, F.lag(col, lag).over(window_spec))
        
        return result_df
    
    def create_rolling_features(self, df, id_cols, target_cols, windows=None):
        """
        Crée des caractéristiques rolling (moyennes mobiles, etc.) pour les colonnes cibles.
        
        Args:
            df: DataFrame Spark avec les données temporelles
            id_cols: Liste des colonnes d'identification (e.g., 'channel')
            target_cols: Liste des colonnes pour lesquelles créer des rolling features
            windows: Liste des tailles de fenêtre en jours
            
        Returns:
            DataFrame Spark avec les caractéristiques de moyenne mobile
        """
        if windows is None:
            windows = self.config['feature_engineering'].get('rolling_windows', [7, 14, 30])
        
        result_df = df
        
        # Pour chaque colonne cible
        for col in target_cols:
            # Définir une fenêtre partitionnée par les colonnes d'ID et ordonnée par date
            for window_size in windows:
                window_spec = Window.partitionBy(id_cols).orderBy('date').rowsBetween(-window_size, 0)
                
                # Créer des caractéristiques d'agrégation sur la fenêtre
                avg_col_name = f"{col}_avg_{window_size}d"
                sum_col_name = f"{col}_sum_{window_size}d"
                max_col_name = f"{col}_max_{window_size}d"
                
                result_df = result_df.withColumn(avg_col_name, F.avg(col).over(window_spec))
                result_df = result_df.withColumn(sum_col_name, F.sum(col).over(window_spec))
                result_df = result_df.withColumn(max_col_name, F.max(col).over(window_spec))
        
        return result_df
    
    def create_seasonality_features(self, df):
        """
        Crée des caractéristiques de saisonnalité basées sur la date.
        
        Args:
            df: DataFrame Spark avec une colonne de date
            
        Returns:
            DataFrame avec des caractéristiques saisonnières ajoutées
        """
        # Vérifier si la fonctionnalité est activée
        if not self.config['feature_engineering'].get('create_seasonality_features', True):
            return df
        
        # Extraire des caractéristiques de date
        result_df = df.withColumn('day_of_week', F.dayofweek('date'))
        result_df = result_df.withColumn('day_of_month', F.dayofmonth('date'))
        result_df = result_df.withColumn('month', F.month('date'))
        result_df = result_df.withColumn('quarter', F.quarter('date'))
        result_df = result_df.withColumn('year', F.year('date'))
        result_df = result_df.withColumn('is_weekend', F.when((F.col('day_of_week') == 1) | (F.col('day_of_week') == 7), 1).otherwise(0))
        
        # Créer des encodages cycliques pour les caractéristiques périodiques
        result_df = result_df.withColumn('day_of_week_sin', F.sin(2 * 3.14159 * F.col('day_of_week') / 7))
        result_df = result_df.withColumn('day_of_week_cos', F.cos(2 * 3.14159 * F.col('day_of_week') / 7))
        result_df = result_df.withColumn('month_sin', F.sin(2 * 3.14159 * F.col('month') / 12))
        result_df = result_df.withColumn('month_cos', F.cos(2 * 3.14159 * F.col('month') / 12))
        
        return result_df
    
    def create_holiday_features(self, df, country='FR'):
        """
        Ajoute des caractéristiques pour les jours fériés.
        
        Args:
            df: DataFrame Spark avec une colonne de date
            country: Code pays pour les jours fériés
            
        Returns:
            DataFrame avec des indicateurs de jours fériés
        """
        # Vérifier si la fonctionnalité est activée
        if not self.config['feature_engineering'].get('create_holiday_features', True):
            return df
        
        # Liste des jours fériés français (exemple simplifié)
        french_holidays = [
            ('01-01', 'New Year'),
            ('05-01', 'Labor Day'),
            ('05-08', 'Victory Day'),
            ('07-14', 'Bastille Day'),
            ('08-15', 'Assumption Day'),
            ('11-01', 'All Saints Day'),
            ('11-11', 'Armistice Day'),
            ('12-25', 'Christmas Day')
        ]
        
        result_df = df
        
        # Ajouter un indicateur pour chaque jour férié
        for date_pattern, holiday_name in french_holidays:
            month, day = date_pattern.split('-')
            result_df = result_df.withColumn(
                f'is_{holiday_name.lower().replace(" ", "_")}',
                F.when(
                    (F.month('date') == int(month)) & 
                    (F.dayofmonth('date') == int(day)),
                    1
                ).otherwise(0)
            )
        
        # Ajouter un indicateur global de jour férié
        holiday_columns = [f'is_{holiday_name.lower().replace(" ", "_")}' for _, holiday_name in french_holidays]
        holiday_sum_expr = '+'.join(holiday_columns)
        result_df = result_df.withColumn('is_holiday', F.expr(f'CASE WHEN ({holiday_sum_expr}) > 0 THEN 1 ELSE 0 END'))
        
        return result_df
