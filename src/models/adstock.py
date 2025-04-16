from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, lag, expr
import pandas as pd
import numpy as np
import math

class AdstockModels:
    """
    Classe implémentant différents modèles d'adstock pour le MMM.
    Ces modèles permettent de capturer les effets retardés des investissements marketing.
    """
    
    @staticmethod
    def geometric_adstock(df, spend_col, decay_rate=0.7, max_lag=10):
        """
        Applique un modèle d'adstock géométrique.
        
        Args:
            df: DataFrame pandas avec les données
            spend_col: Nom de la colonne de dépense
            decay_rate: Taux de décroissance (0 à 1)
            max_lag: Nombre maximum de périodes pour l'effet retardé
            
        Returns:
            Série pandas avec les valeurs d'adstock
        """
        # Vérifier que le DataFrame est trié par date
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise ValueError("Le DataFrame doit avoir un index de type datetime")
        
        # Initialiser la série d'adstock
        adstock = pd.Series(0, index=df.index)
        spend = df[spend_col]
        
        # Calculer l'adstock
        for t in range(len(df)):
            # Effet immédiat
            adstock.iloc[t] += spend.iloc[t]
            
            # Effets retardés
            for lag in range(1, min(max_lag, len(df) - t)):
                adstock.iloc[t + lag] += spend.iloc[t] * (decay_rate ** lag)
        
        return adstock
    
    @staticmethod
    def weibull_adstock(df, spend_col, shape=2.0, scale=7.0, max_lag=30):
        """
        Applique un modèle d'adstock Weibull, plus flexible que le géométrique.
        
        Args:
            df: DataFrame pandas avec les données
            spend_col: Nom de la colonne de dépense
            shape: Paramètre de forme (> 0)
            scale: Paramètre d'échelle (> 0)
            max_lag: Nombre maximum de périodes pour l'effet retardé
            
        Returns:
            Série pandas avec les valeurs d'adstock
        """
        # Vérifier que le DataFrame est trié par date
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            raise ValueError("Le DataFrame doit avoir un index de type datetime")
        
        # Calculer les poids Weibull normalisés
        lags = np.arange(max_lag)
        weights = (shape / scale) * ((lags / scale) ** (shape - 1)) * np.exp(-((lags / scale) ** shape))
        weights = weights / weights.sum()  # Normaliser
        
        # Initialiser la série d'adstock
        adstock = pd.Series(0, index=df.index)
        spend = df[spend_col]
        
        # Calculer l'adstock
        for t in range(len(df)):
            # Appliquer les poids à tous les lags
            for lag in range(min(max_lag, len(df) - t)):
                adstock.iloc[t + lag] += spend.iloc[t] * weights[lag]
        
        return adstock
    
    @staticmethod
    def apply_saturation(x, saturation_type='hill', k=0.5, S=10000):
        """
        Applique une fonction de saturation aux valeurs d'adstock.
        
        Args:
            x: Valeur d'adstock
            saturation_type: Type de fonction ('hill', 'exp', 'logistic')
            k: Paramètre de saturation
            S: Point de diminution des rendements marginaux
            
        Returns:
            Valeur transformée
        """
        if saturation_type == 'hill':
            # Fonction de Hill (utilisée dans le papier de Google/Meta)
            return x ** k / (x ** k + S ** k)
        
        elif saturation_type == 'exp':
            # Fonction exponentielle
            return 1 - np.exp(-k * x / S)
        
        elif saturation_type == 'logistic':
            # Fonction logistique
            return 1 / (1 + np.exp(-k * (x - S)))
        
        else:
            # Identité (pas de saturation)
            return x
    
    @staticmethod
    def process_channel_spark(spark_df, channel, params, date_col='date'):
        """
        Traite un canal avec effets d'adstock et de saturation en utilisant Spark.
        
        Args:
            spark_df: DataFrame Spark
            channel: Nom du canal
            params: Dictionnaire de paramètres (decay, lag, saturation)
            date_col: Nom de la colonne de date
            
        Returns:
            DataFrame Spark avec la colonne adstock ajoutée
        """
        # Convertir en DataFrame pandas pour les calculs
        pdf = spark_df.select(date_col, channel).toPandas()
        pdf[date_col] = pd.to_datetime(pdf[date_col])
        pdf.set_index(date_col, inplace=True)
        pdf.sort_index(inplace=True)
        
        # Appliquer le modèle d'adstock
        if params.get('adstock_type') == 'weibull':
            adstock = AdstockModels.weibull_adstock(
                pdf, 
                channel,
                shape=params.get('shape', 2.0),
                scale=params.get('scale', 7.0),
                max_lag=params.get('max_lag', 30)
            )
        else:
            # Par défaut, utiliser l'adstock géométrique
            adstock = AdstockModels.geometric_adstock(
                pdf, 
                channel,
                decay_rate=params.get('decay_rate', 0.7),
                max_lag=params.get('max_lag', 10)
            )
        
        # Appliquer la saturation
        transformed = AdstockModels.apply_saturation(
            adstock,
            saturation_type=params.get('saturation_type', 'hill'),
            k=params.get('k', 0.5),
            S=params.get('S', 10000)
        )
        
        # Créer un nouveau DataFrame avec la date et la transformation
        result_pdf = pd.DataFrame({
            date_col: pdf.index,
            f"{channel}_adstock": transformed
        })
        
        # Convertir en DataFrame Spark
        result_df = spark_df.sparkSession.createDataFrame(result_pdf)
        result_df = result_df.withColumn(date_col, col(date_col).cast("date"))
        
        # Joindre avec le DataFrame original
        return spark_df.join(result_df, on=date_col)
