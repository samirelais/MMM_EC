import pytest
import os
import sys
import pandas as pd
from datetime import datetime
from pyspark.sql import SparkSession

# Ajouter le dossier parent au chemin Python
sys.path.append(os.path.abspath(".."))

# Importer conditionnellement pour éviter les erreurs dans les CI/CD
try:
    from src.data.online_retail_loader import OnlineRetailLoader
except ImportError:
    pass

# Fixture pour la session Spark
@pytest.fixture(scope="module")
def spark():
    """Crée une session Spark pour les tests."""
    spark = SparkSession.builder \
        .appName("mmm_test") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()

# Fixture pour la configuration de test
@pytest.fixture(scope="module")
def test_config():
    """Crée une configuration de test."""
    return {
        "data": {
            "retail_data_path": "tests/data/test_retail.csv",
            "include_returns": False,
            "start_date": "2010-12-01",
            "end_date": "2011-12-09"
        },
        "marketing_channels": [
            "tv", "radio", "print", "social_media", "search", "email", "display"
        ]
    }

# Fixture pour les données de test
@pytest.fixture(scope="module")
def test_data(spark):
    """Crée des données de test."""
    # Données retail de test
    retail_data = [
        (1, "2010-12-01", "A", 10, 2.5, "C1", "UK"),
        (2, "2010-12-01", "B", 5, 3.0, "C2", "FR"),
        (3, "2010-12-02", "A", -2, 2.5, "C1", "UK"),  # Retour
        (4, "2010-12-02", "C", 8, 1.0, "C3", "DE")
    ]
    
    # Créer le DataFrame
    retail_df = spark.createDataFrame(
        retail_data,
        ["InvoiceNo", "InvoiceDate", "Description", "Quantity", "UnitPrice", "CustomerID", "Country"]
    )
    
    # Convertir les colonnes appropriées
    retail_df = retail_df.withColumn("InvoiceDate", retail_df["InvoiceDate"].cast("date"))
    
    return retail_df

# Test d'initialisation
def test_loader_init(spark, test_config):
    """Teste l'initialisation du loader."""
    # Créer un fichier de configuration temporaire
    import json
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        config_path = f.name
    
    try:
        # Initialiser le loader
        loader = OnlineRetailLoader(spark, config_path)
        
        # Vérifier que les attributs sont correctement initialisés
        assert loader.spark == spark
        assert loader.config == test_config
    finally:
        # Supprimer le fichier temporaire
        os.unlink(config_path)

# Test de création des données de ventes quotidiennes
def test_create_daily_sales_data(spark, test_config, test_data):
    """Teste la création des données de ventes quotidiennes."""
    # Créer un fichier de configuration temporaire
    import json
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_config, f)
        config_path = f.name
    
    try:
        # Initialiser le loader
        loader = OnlineRetailLoader(spark, config_path)
        
        # Créer des données de ventes quotidiennes
        daily_sales = loader.create_daily_sales_data(test_data)
        
        # Vérifier les résultats
        assert daily_sales.count() == 2  # 2 jours distincts
        
        # Convertir en pandas pour des vérifications plus faciles
        pdf = daily_sales.toPandas()
        
        # Vérifier que les colonnes attendues sont présentes
        assert all(col in pdf.columns for col in ['date', 'revenue', 'transactions', 'unique_customers'])
        
        # Vérifier les valeurs pour un jour spécifique
        day1 = pdf[pdf['date'] == datetime(2010, 12, 1).date()]
        assert len(day1) == 1
        assert day1['revenue'].values[0] == (10 * 2.5 + 5 * 3.0)  # 25 + 15 = 40
        assert day1['transactions'].values[0] == 2
        assert day1['unique_customers'].values[0] == 2
    finally:
        # Supprimer le fichier temporaire
        os.unlink(config_path)
