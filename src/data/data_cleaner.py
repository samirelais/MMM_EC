from pyspark.sql import functions as F
from pyspark.sql.window import Window
import json

class DataCleaner:
    """
    Classe pour nettoyer et prétraiter les données avant la modélisation.
    """

    def __init__(self, config_path):
        """
        Initialise le DataCleaner avec un chemin de configuration.

        Args:
            config_path: Chemin vers le fichier de configuration JSON
        """
        # Charger la configuration
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)

    def remove_outliers(self, df, column, threshold=None):
        """
        Supprime les valeurs aberrantes d'une colonne spécifique en utilisant la méthode IQR.

        Args:
            df: DataFrame Spark à nettoyer
            column: Nom de la colonne à nettoyer
            threshold: Seuil pour la détection des valeurs aberrantes (par défaut, utilise la config)

        Returns:
            DataFrame Spark nettoyé
        """
        if threshold is None:
            threshold = self.config['preprocessing'].get('outlier_threshold', 3)

        # Calculer les quantiles pour la détection des valeurs aberrantes
        quantiles = df.approxQuantile(column, [0.25, 0.5, 0.75], 0.05)
        q1, median, q3 = quantiles

        # Calculer l'IQR et les limites
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        # Filtrer les valeurs aberrantes
        df_clean = df.filter(
            (F.col(column) >= lower_bound) &
            (F.col(column) <= upper_bound)
        )

        # Compter le nombre de lignes supprimées
        count_before = df.count()
        count_after = df_clean.count()
        print(f"Suppression de {count_before - count_after} valeurs aberrantes dans la colonne {column}")

        return df_clean

    def fill_missing_values(self, df, method=None):
        """
        Remplit les valeurs manquantes dans le DataFrame.

        Args:
            df: DataFrame Spark à nettoyer
            method: Méthode de remplissage ('mean', 'median', 'interpolate', etc.)

        Returns:
            DataFrame Spark sans valeurs manquantes
        """
        if method is None:
            method = self.config['preprocessing'].get('fill_missing_strategy', 'interpolate')

        # Obtenir la liste des colonnes numériques
        numeric_cols = [f.name for f in df.schema.fields if f.dataType.simpleString() in ['double', 'int', 'float']]

        # Traiter selon la méthode choisie
        if method == 'mean':
            # Calculer les moyennes pour chaque colonne
            means = {col: df.select(F.mean(col)).collect()[0][0] for col in numeric_cols}

            # Remplir avec les moyennes
            for col in numeric_cols:
                df = df.fillna(means[col], subset=[col])

        elif method == 'median':
            # Calculer les médianes pour chaque colonne
            medians = {col: df.approxQuantile(col, [0.5], 0.05)[0] for col in numeric_cols}

            # Remplir avec les médianes
            for col in numeric_cols:
                df = df.fillna(medians[col], subset=[col])

        elif method == 'interpolate':
            # Utiliser une fenêtre glissante pour interpoler
            for col in numeric_cols:
                # Définir la fenêtre temporelle ordonnée par date
                w = Window.orderBy('date')

                # Appliquer une interpolation linéaire
                df = df.withColumn(
                    col,
                    F.when(F.col(col).isNull(),
                          (F.last(col, True).over(w.rowsBetween(-sys.maxsize, -1)) +
                           F.first(col, True).over(w.rowsBetween(1, sys.maxsize))) / 2
                    ).otherwise(F.col(col))
                )

        return df
