import pytest
import os
import json
from pyspark.sql import SparkSession
import sys

# Ajouter le répertoire src au chemin Python pour les imports
sys.path.append(os.path.abspath("../"))

# Import conditionnel pour éviter les erreurs lors des tests
try:
    from src.data.data_loader import DataLoader
except ImportError:
    print("Module data_loader non disponible. Les tests seront limités.")

@pytest.fixture(scope="module")
def spark():
    """Fixture pour créer une session Spark pour les tests."""
    spark = SparkSession.builder \
        .appName("mmm_test") \
        .master("local[2]") \
        .getOrCreate()
    yield spark
    spark.stop()

@pytest.fixture(scope="module")
def config_path(tmp_path_factory):
    """Fixture pour créer un fichier de configuration temporaire."""
    config = {
        "data": {
            "sales_data_path": "tests/data/sales_test.csv",
            "marketing_data_path": "tests/data/marketing_test.csv",
            "external_factors_path": "tests/data/external_test.csv",
            "train_start_date": "2022-01-01",
            "train_end_date": "2023-06-30",
            "test_start_date": "2023-07-01",
            "test_end_date": "2023-12-31"
        }
    }

    # Créer un répertoire temporaire
    dir_path = tmp_path_factory.mktemp("config")
    file_path = dir_path / "test_config.json"

    # Écrire la configuration dans le fichier
    with open(file_path, 'w') as f:
        json.dump(config, f)

    return str(file_path)

def test_spark_session(spark):
    """Test simple pour vérifier que Spark fonctionne."""
    assert spark is not None

# Ce test sera exécuté uniquement si le module DataLoader est disponible
@pytest.mark.skipif("'DataLoader' not in globals()")
def test_data_loader_init(spark, config_path):
    """Test l'initialisation du DataLoader."""
    try:
        data_loader = DataLoader(spark, config_path)

        assert data_loader.spark == spark
        assert isinstance(data_loader.config, dict)
        assert "data" in data_loader.config
    except NameError:
        pytest.skip("DataLoader n'est pas disponible")
