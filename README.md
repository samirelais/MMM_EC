# Marketing Mix Modeling (MMM) avec PySpark

Ce projet implémente une analyse de Marketing Mix Modeling (MMM) complète avec PySpark pour optimiser l'allocation budgétaire marketing. Il est structuré comme une solution production-ready qui peut être appliquée à de multiples jeux de données.

## Objectifs
- Évaluer l'impact de différents canaux marketing sur les ventes/revenus
- Quantifier le retour sur investissement (ROI) par canal
- Optimiser l'allocation budgétaire pour maximiser les revenus
- Fournir des recommandations stratégiques basées sur des données

## Fonctionnalités
- **Pipeline de données complet** : Nettoyage, transformation et préparation des données pour le modeling
- **Features engineering avancé** : Création de caractéristiques temporelles, saisonnières et de retard (lag features)
- **Modélisation robuste** : Utilisation de LightGBM pour la modélisation avancée
- **Optimisation budgétaire** : Algorithmes d'optimisation pour maximiser le ROI
- **Visualisations interactives** : Dashboard Streamlit pour explorer les résultats
- **Génération de rapports** : Rapports automatiques en Markdown et HTML
- **Architecture scalable** : Conçu pour gérer de grands volumes de données avec PySpark

## Comment utiliser ce projet

1. Cloner le dépôt
2. Installer les dépendances: `pip install -r requirements.txt`
3. Configurer les paramètres dans `config/mmm_config.json`
4. Exécuter le pipeline complet: `python scripts/run_pipeline.py`
5. Consulter les résultats dans le dossier `reports/`
