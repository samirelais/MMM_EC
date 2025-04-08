from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, unix_timestamp, dayofmonth, month, year, sum, count, when, lit
from pyspark.sql.window import Window
import os
import json

class OnlineRetailLoader:
    """
    Classe pour charger et transformer les données Online Retail pour l'analyse MMM.
    """
    
    def __init__(self, spark_session, config_path):
        """
        Initialise le loader avec une session Spark et un chemin de configuration.
        
        Args:
            spark_session: Session Spark active
            config_path: Chemin vers le fichier de configuration JSON
        """
        self.spark = spark_session
        
        # Charger la configuration
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
    
    def load_retail_data(self):
        """
        Charge les données Online Retail et effectue un prétraitement de base.
        
        Returns:
            DataFrame Spark contenant les données nettoyées
        """
        # Définir le chemin du fichier
        data_path = self.config['data'].get('retail_data_path', 'data/online_retail.csv')
        
        # Charger les données
        print(f"Chargement des données depuis {data_path}...")
        retail_df = self.spark.read.csv(data_path, header=True, inferSchema=True)
        
        # Convertir la colonne de date et créer des colonnes de date formatées
        print("Conversion des dates et nettoyage des données...")
        retail_df = retail_df.withColumn("InvoiceDate", to_date(col("InvoiceDate")))
        
        # Calculer le montant total par transaction (Quantity * UnitPrice)
        retail_df = retail_df.withColumn("TotalAmount", col("Quantity") * col("UnitPrice"))
        
        # Filtrer les lignes avec des quantités négatives (retours) si spécifié dans la config
        if not self.config['data'].get('include_returns', True):
            retail_df = retail_df.filter(col("Quantity") > 0)
        
        # Filtrer par plage de dates si spécifié
        start_date = self.config['data'].get('start_date')
        end_date = self.config['data'].get('end_date')
        
        if start_date and end_date:
            retail_df = retail_df.filter(
                (col("InvoiceDate") >= start_date) & 
                (col("InvoiceDate") <= end_date)
            )
        
        print(f"Données chargées avec succès : {retail_df.count()} lignes.")
        return retail_df
    
    def create_daily_sales_data(self, retail_df):
        """
        Agrège les données au niveau journalier pour l'analyse MMM.
        
        Args:
            retail_df: DataFrame Spark contenant les données retail brutes
            
        Returns:
            DataFrame Spark avec les ventes quotidiennes
        """
        print("Création des données de ventes quotidiennes...")
        
        # Grouper par date
        daily_sales = retail_df.groupBy("InvoiceDate").agg(
            sum("TotalAmount").alias("revenue"),
            count("InvoiceNo").alias("transactions"),
            count(when(col("CustomerID").isNotNull(), True)).alias("unique_customers")
        )
        
        # Renommer la colonne de date pour la cohérence avec notre modèle
        daily_sales = daily_sales.withColumnRenamed("InvoiceDate", "date")
        
        # Trier par date
        daily_sales = daily_sales.orderBy("date")
        
        print(f"Données de ventes quotidiennes créées : {daily_sales.count()} jours.")
        return daily_sales
    
    def create_marketing_channel_data(self, retail_df):
        """
        Crée des données marketing simulées basées sur les ventes réelles.
        Pour un jeu de données réel, nous n'avons pas les dépenses marketing,
        donc nous les simulons en fonction des patterns de vente.
        
        Args:
            retail_df: DataFrame Spark contenant les données retail brutes
            
        Returns:
            DataFrame Spark contenant les dépenses marketing simulées par canal et par jour
        """
        print("Création des données marketing simulées basées sur les patterns de vente...")
        
        # D'abord, obtenir les ventes quotidiennes
        daily_sales = self.create_daily_sales_data(retail_df)
        
        # Liste des canaux marketing à simuler
        channels = self.config['marketing_channels']
        
        # Créer un DataFrame de base avec toutes les dates
        date_df = daily_sales.select("date").distinct()
        
        # Convertir en pandas pour faciliter la simulation
        pdf = date_df.toPandas()
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.sort_values("date")
        
        # Simuler les dépenses pour chaque canal
        for i, channel in enumerate(channels):
            # Chaque canal a une distribution différente
            base_spend = 1000 * (i + 1)  # Budget de base différent par canal
            seasonal_factor = 0.3 * np.sin(2 * np.pi * (pdf["date"].dt.month - i) / 12) + 1  # Saisonnalité
            weekly_factor = 0.2 * np.sin(2 * np.pi * (pdf["date"].dt.dayofweek) / 7) + 1  # Variation hebdomadaire
            
            # Ajouter un facteur de tendance spécifique au canal
            trend_factor = 1 + 0.001 * (i % 3) * np.arange(len(pdf))  # Tendance variable par canal
            
            # Calculer les dépenses
            pdf[channel] = base_spend * seasonal_factor * weekly_factor * trend_factor
            
            # Ajouter du bruit aléatoire
            pdf[channel] = pdf[channel] * (0.9 + 0.2 * np.random.random(len(pdf)))
            
            # Arrondir à 2 décimales
            pdf[channel] = pdf[channel].round(2)
        
        # Mettre en forme pour le format MMM (long format)
        channels_data = []
        for channel in channels:
            channel_pdf = pdf[["date", channel]].copy()
            channel_pdf["channel"] = channel
            channel_pdf["spend"] = channel_pdf[channel]
            channel_pdf = channel_pdf[["date", "channel", "spend"]]
            channels_data.append(channel_pdf)
        
        # Combiner tous les canaux
        marketing_pdf = pd.concat(channels_data, ignore_index=True)
        
        # Convertir en DataFrame Spark
        marketing_df = self.spark.createDataFrame(marketing_pdf)
        
        print(f"Données marketing simulées créées : {marketing_df.count()} lignes.")
        return marketing_df
    
    def create_external_factors(self, retail_df):
        """
        Crée des facteurs externes simulés (économie, événements, etc.)
        basés sur les patterns de vente réels.
        
        Args:
            retail_df: DataFrame Spark contenant les données retail brutes
            
        Returns:
            DataFrame Spark contenant les facteurs externes simulés
        """
        print("Création des facteurs externes simulés...")
        
        # Obtenir les dates uniques
        daily_sales = self.create_daily_sales_data(retail_df)
        date_df = daily_sales.select("date").distinct()
        
        # Convertir en pandas pour faciliter la simulation
        pdf = date_df.toPandas()
        pdf["date"] = pd.to_datetime(pdf["date"])
        pdf = pdf.sort_values("date")
        
        # Simuler les indices économiques
        # Indice de confiance des consommateurs
        base_confidence = 100
        pdf["consumer_confidence"] = base_confidence + \
                                    5 * np.sin(2 * np.pi * pdf["date"].dt.dayofyear / 365) + \
                                    np.random.normal(0, 1, len(pdf))
        
        # Croissance du PIB (trimestrielle, mais interpolée quotidiennement)
        quarterly_growth = np.array([2.1, 2.3, 2.0, 1.9, 2.2, 2.4, 2.5, 2.3])
        quarters = np.floor(pdf["date"].dt.month / 3).astype(int) % len(quarterly_growth)
        pdf["gdp_growth"] = quarterly_growth[quarters] + np.random.normal(0, 0.1, len(pdf))
        
        # Taux de chômage
        pdf["unemployment"] = 5.0 + 0.3 * np.sin(2 * np.pi * pdf["date"].dt.dayofyear / 365) + \
                             np.random.normal(0, 0.05, len(pdf))
        
        # Événements spéciaux (vacances, promotions, etc.)
        # 1 si c'est un jour férié/événement spécial, 0 sinon
        # Liste simplifiée des jours fériés au Royaume-Uni
        uk_holidays = [
            "2010-01-01", "2010-04-02", "2010-04-05", "2010-05-03", "2010-05-31", 
            "2010-08-30", "2010-12-25", "2010-12-26", "2010-12-27", "2010-12-28",
            "2011-01-01", "2011-04-22", "2011-04-25", "2011-05-02", "2011-05-30",
            "2011-08-29", "2011-12-25", "2011-12-26", "2011-12-27"
        ]
        pdf["is_holiday"] = pdf["date"].dt.strftime("%Y-%m-%d").isin(uk_holidays).astype(int)
        
        # Événements promotionnels simulés (Black Friday, Boxing Day, etc.)
        promo_days = [
            "2010-11-26", "2010-12-26", "2011-01-01", "2011-07-01", "2011-11-25", "2011-12-26"
        ]
        pdf["is_promo_event"] = pdf["date"].dt.strftime("%Y-%m-%d").isin(promo_days).astype(int)
        
        # Convertir en DataFrame Spark
        external_df = self.spark.createDataFrame(pdf)
        
        print(f"Facteurs externes simulés créés : {external_df.count()} lignes.")
        return external_df
