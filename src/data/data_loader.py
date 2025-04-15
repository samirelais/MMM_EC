from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date
import json
import os

class DataLoader:
    """
    Classe pour charger les données à partir de différentes sources
    et les préparer pour l'analyse.
    """

    def __init__(self, spark_session, config_path):
        """
        Initialise le DataLoader avec une session Spark et un chemin de configuration.

        Args:
            spark_session: Session Spark active
            config_path: Chemin vers le fichier de configuration JSON
        """
        self.spark = spark_session

        # Charger la configuration
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

    def load_sales_data(self):
        """
        Charge les données de vente et effectue un prétraitement de base.

        Returns:
            DataFrame Spark contenant les données de vente nettoyées
        """
        sales_path = self.config['data']['sales_data_path']

        # Charger les données depuis CSV ou autre source
        sales_df = self.spark.read.csv(sales_path, header=True, inferSchema=True)

        # Convertir la colonne de date
        sales_df = sales_df.withColumn("date", to_date(col("date")))

        # Filtrer par plage de dates si spécifié
        train_start = self.config['data']['train_start_date']
        test_end = self.config['data']['test_end_date']

        sales_df = sales_df.filter(
            (col("date") >= train_start) &
            (col("date") <= test_end)
        )

        return sales_df

    def load_marketing_data(self):
        """
        Charge les données marketing et effectue un prétraitement de base.

        Returns:
            DataFrame Spark contenant les données marketing nettoyées
        """
        marketing_path = self.config['data']['marketing_data_path']

        # Charger les données depuis CSV ou autre source
        marketing_df = self.spark.read.csv(marketing_path, header=True, inferSchema=True)

        # Convertir la colonne de date
        marketing_df = marketing_df.withColumn("date", to_date(col("date")))

        # Filtrer par plage de dates si spécifié
        train_start = self.config['data']['train_start_date']
        test_end = self.config['data']['test_end_date']

        marketing_df = marketing_df.filter(
            (col("date") >= train_start) &
            (col("date") <= test_end)
        )

        return marketing_df

    def load_external_factors(self):
        """
        Charge les facteurs externes comme les données macroéconomiques.

        Returns:
            DataFrame Spark contenant les facteurs externes
        """
        external_path = self.config['data']['external_factors_path']

        # Vérifier si le fichier existe
        if not os.path.exists(external_path):
            print(f"Le fichier {external_path} n'existe pas. Retour d'un DataFrame vide.")
            schema = "date DATE, gdp DOUBLE, unemployment DOUBLE, consumer_confidence DOUBLE"
            return self.spark.createDataFrame([], schema)

        # Charger les données depuis CSV ou autre source
        external_df = self.spark.read.csv(external_path, header=True, inferSchema=True)

        # Convertir la colonne de date
        external_df = external_df.withColumn("date", to_date(col("date")))

        return external_df
