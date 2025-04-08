```markdown
# Marketing Mix Modeling (MMM) avec PySpark pour E-commerce

Ce projet implémente une analyse de Marketing Mix Modeling (MMM) complète avec PySpark pour optimiser l'allocation budgétaire marketing d'un détaillant en ligne. Il utilise les données Open Source "Online Retail" de Kaggle et est structuré comme une solution prête pour la production.

## Objectifs

- Évaluer l'impact de différents canaux marketing sur les ventes/revenus
- Quantifier le retour sur investissement (ROI) par canal
- Optimiser l'allocation budgétaire pour maximiser les revenus
- Fournir des recommandations stratégiques basées sur les données

## Fonctionnalités

- **Pipeline de données complet** : Nettoyage, transformation et préparation des données pour le modeling
- **Features engineering avancé** : Création de caractéristiques temporelles, saisonnières et de retard (lag features)
- **Modélisation robuste** : Utilisation de LightGBM pour la modélisation avancée
- **Transformation d'adstock** : Modélisation des effets retardés des investissements marketing
- **Optimisation budgétaire** : Algorithmes d'optimisation pour maximiser le ROI
- **Visualisations interactives** : Dashboard Streamlit pour explorer les résultats
- **Génération de rapports** : Rapports automatiques en Markdown et HTML
- **Architecture scalable** : Conçu pour gérer de grands volumes de données avec PySpark

## Données utilisées

Le projet utilise le jeu de données "Online Retail" disponible sur Kaggle : [https://www.kaggle.com/datasets/vijayuv/onlineretail](https://www.kaggle.com/datasets/vijayuv/onlineretail)

**Caractéristiques des données :**
- Transactions d'un détaillant en ligne basé au Royaume-Uni spécialisé dans les cadeaux
- Période : Décembre 2010 à Décembre 2011
- ~500 000 transactions
- ~4 000 clients uniques
- ~4 000 produits uniques

## Installation et configuration

### Prérequis

- Python 3.8+
- PySpark 3.x
- Compte Kaggle (pour télécharger les données)

### Installation

1. Cloner le dépôt :
```bash
git clone https://github.com/samirelais/MMM_ECOMMERCE.git
cd MMM_ECOMMERCE
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Initialiser le projet et télécharger les données :
```bash
python scripts/init_project.py
```

## Structure du projet

```
mmm-ecommerce/
├── README.md                       # Documentation principale
├── requirements.txt                # Dépendances Python
├── config/                         # Configurations
│   └── online_retail_config.json   # Configuration pour Online Retail
├── data/                           # Données
├── notebooks/                      # Notebooks d'exploration
│   └── 01_online_retail_exploration.ipynb  # Exploration des données
├── src/                            # Code source
│   ├── data/                       # Traitement des données
│   │   ├── data_cleaner.py         # Nettoyage des données
│   │   ├── data_generator.py       # Génération de données simulées
│   │   └── online_retail_loader.py # Chargement des données Online Retail
│   ├── features/                   # Ingénierie des caractéristiques
│   │   └── feature_engineer.py     # Création des caractéristiques
│   ├── models/                     # Modélisation et optimisation
│   │   ├── adstock.py              # Modèles d'adstock
│   │   └── mmm_model.py            # Modèle principal MMM
│   ├── visualization/              # Visualisations
│   │   └── visualization.py        # Génération de graphiques
│   └── pipeline/                   # Pipeline complet
├── tests/                          # Tests unitaires
│   └── test_online_retail_loader.py # Tests du chargeur de données
├── scripts/                        # Scripts d'exécution
│   ├── download_data.py            # Téléchargement des données
│   ├── init_project.py             # Initialisation du projet
│   └── run_online_retail_mmm.py    # Exécution du MMM
├── app/                            # Application web
│   └── mmm_dashboard.py            # Dashboard Streamlit
└── reports/                        # Rapports générés
    └── figures/                    # Graphiques générés
```

## Utilisation

### Exécution de l'analyse MMM

```bash
python scripts/run_online_retail_mmm.py
```

### Lancement du dashboard

```bash
cd app
streamlit run mmm_dashboard.py
```

## Méthodologie MMM

### 1. Prétraitement des données

- Nettoyage et agrégation des données de ventes
- Simulation des dépenses marketing basées sur des patterns réalistes
- Création de facteurs externes (économie, saisonnalité, etc.)

### 2. Feature Engineering

- Création de caractéristiques temporelles (jour de la semaine, mois, etc.)
- Transformation d'adstock pour modéliser les effets retardés
- Modélisation de la saturation pour les rendements décroissants

### 3. Modélisation

- Utilisation de LightGBM pour la modélisation
- Validation croisée temporelle pour éviter le data leakage
- Évaluation sur un ensemble de test chronologique

### 4. Analyse des contributions

- Décomposition des ventes par canal marketing
- Calcul du ROI par canal
- Analyse de l'évolution temporelle des contributions

### 5. Optimisation budgétaire

- Optimisation basée sur le ROI historique
- Prise en compte des contraintes minimales et maximales par canal
- Génération de recommandations d'allocation

## Personnalisation

Pour adapter le modèle à différents contextes :

1. Ajuster les paramètres de configuration dans `config/online_retail_config.json`
2. Modifier les paramètres d'adstock dans `src/models/mmm_model.py`
3. Ajuster les stratégies d'optimisation dans le module MMM

## Tests unitaires

Exécuter les tests unitaires :

```bash
pytest
```

## Documentation

Pour une documentation détaillée sur chaque composant :
- Consultez les docstrings dans le code
- Référez-vous aux notebooks Jupyter dans le répertoire `notebooks/`
- Lisez les rapports générés dans le répertoire `reports/`

## Résultats

L'analyse MMM produit :
- Un rapport détaillé sur les contributions de chaque canal
- Des visualisations des contributions et du ROI
- Des recommandations d'allocation budgétaire optimisée
- Un dashboard interactif pour explorer les résultats

## Licence

Ce projet est sous licence MIT.

## Contributeurs

- Samir Elaissaouy - Développement initial

## Références

- Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017). Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects.
- [Facebook/Meta Robyn](https://github.com/facebookexperimental/Robyn)
- [Google's MMM Python Library](https://github.com/google/lightweight_mmm)
```
