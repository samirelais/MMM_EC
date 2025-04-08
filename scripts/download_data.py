import pandas as pd
import os

# Assurer que le dossier data existe
os.makedirs("data", exist_ok=True)

# Charger le fichier Excel
print("Chargement du fichier Excel...")
df = pd.read_excel('data/Online Retail.xlsx')

# Afficher quelques informations
print(f"Dimensions du dataset: {df.shape}")
print("Premières lignes:")
print(df.head())

# Convertir en CSV
print("Conversion en CSV...")
df.to_csv('data/Online_Retail.csv', index=False)
print("Conversion terminée : data/Online_Retail.csv")