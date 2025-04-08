import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
from datetime import datetime, timedelta

# Ajouter le répertoire parent au chemin Python
sys.path.append(os.path.abspath(".."))

# Importer les modules nécessaires
try:
   from src.data.online_retail_loader import OnlineRetailLoader
   from src.models.mmm_model import MMMModel
   from src.visualization.visualization import MMMVisualization
except ImportError as e:
   st.error(f"Erreur d'importation des modules: {e}")

# Configuration de la page
st.set_page_config(
   page_title="Dashboard Marketing Mix Modeling",
   page_icon="📊",
   layout="wide",
   initial_sidebar_state="expanded"
)

# Titre de l'application
st.title("📊 Dashboard Marketing Mix Modeling")
st.write("Analyse et optimisation de l'attribution marketing basée sur les données Online Retail")

# Fonction pour charger la configuration
@st.cache_data
def load_config():
   with open("../config/online_retail_config.json", "r") as f:
       return json.load(f)

# Fonction pour charger les résultats du modèle
@st.cache_data
def load_results():
   try:
       # Charger les contributions
       contributions_df = pd.read_csv("../reports/channel_contributions.csv")
       
       # Charger l'allocation budgétaire
       budget_df = pd.read_csv("../reports/budget_allocation.csv")
       
       # Charger les métriques
       with open("../reports/model_metrics.json", "r") as f:
           metrics = json.load(f)
       
       return contributions_df, budget_df, metrics
   except FileNotFoundError:
       st.warning("Résultats du modèle non trouvés. Veuillez d'abord exécuter le script d'analyse MMM.")
       return None, None, None

# Charger la configuration
config = load_config()

# Sidebar pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
   "Choisir une page",
   ["Vue d'ensemble", "Analyse des contributions", "Allocation budgétaire", "Simulateur", "À propos"]
)

# Sidebar pour les informations
st.sidebar.title("Informations")
st.sidebar.info(
   """
   Ce dashboard présente les résultats d'une analyse 
   de Marketing Mix Modeling (MMM) sur les données 
   Online Retail.
   
   **Source des données:** 
   [Kaggle - Online Retail Dataset](https://www.kaggle.com/datasets/vijayuv/onlineretail)
   """
)

# Tenter de charger les résultats
contributions_df, budget_df, metrics = load_results()

if page == "Vue d'ensemble":
   st.header("Vue d'ensemble du projet MMM")
   
   col1, col2 = st.columns([2, 1])
   
   with col1:
       st.subheader("Qu'est-ce que le Marketing Mix Modeling?")
       st.write(
           """
           Le Marketing Mix Modeling (MMM) est une technique statistique qui analyse 
           l'impact des différentes activités marketing sur les ventes. Elle permet de :
           
           - **Quantifier** l'efficacité de chaque canal marketing
           - **Mesurer** le retour sur investissement (ROI)
           - **Optimiser** l'allocation budgétaire
           - **Prévoir** l'impact des futurs investissements marketing
           """
       )
       
       st.subheader("À propos des données Online Retail")
       st.write(
           """
           Les données utilisées pour cette analyse proviennent du jeu de données 
           "Online Retail" disponible sur Kaggle. Il s'agit de données transactionnelles 
           d'un détaillant en ligne basé au Royaume-Uni, spécialisé dans les cadeaux.
           
           **Caractéristiques des données:**
           - Période: Décembre 2010 à Décembre 2011
           - ~500 000 transactions
           - ~4 000 clients uniques
           - ~4 000 produits uniques
           """
       )
   
   with col2:
       st.image("https://miro.medium.com/max/1200/1*6JgtnS-nglYMXmh32xqYxQ.png", 
               caption="Exemple de Marketing Mix Modeling")
   
   # Afficher les métriques du modèle si disponibles
   if metrics:
       st.subheader("Performance du modèle")
       
       col1, col2, col3, col4 = st.columns(4)
       
       with col1:
           st.metric(label="R²", value=f"{metrics['r2']:.3f}")
       
       with col2:
           st.metric(label="RMSE", value=f"{metrics['rmse']:.2f}")
       
       with col3:
           st.metric(label="MAE", value=f"{metrics['mae']:.2f}")
       
       with col4:
           st.metric(label="MAPE", value=f"{metrics['mape']:.2f}%")
   
   # Graphique des ventes au fil du temps
   st.subheader("Évolution des ventes")
   if contributions_df is not None:
       fig, ax = plt.subplots(figsize=(12, 6))
       
       # Convertir les dates
       contributions_df['date'] = pd.to_datetime(contributions_df['date'])
       contributions_df = contributions_df.sort_values('date')
       
       # Tracer les ventes réelles et prédites
       ax.plot(contributions_df['date'], contributions_df['actual_revenue'], 
               label='Ventes réelles', color='blue')
       ax.plot(contributions_df['date'], contributions_df['predicted_revenue'], 
               label='Ventes prédites', color='red', linestyle='--')
       
       # Formatage
       ax.set_xlabel('Date')
       ax.set_ylabel('Ventes (£)')
       ax.set_title('Évolution des ventes - Réelles vs Prédites')
       ax.legend()
       ax.grid(True, alpha=0.3)
       plt.xticks(rotation=45)
       plt.tight_layout()
       
       st.pyplot(fig)
   else:
       st.info("Graphique non disponible. Veuillez d'abord exécuter l'analyse MMM.")

elif page == "Analyse des contributions":
   st.header("Analyse des contributions par canal")
   
   if contributions_df is not None:
       # Préparer les données
       channels = config['marketing_channels']
       contrib_cols = [f"{ch}_contribution" for ch in channels if f"{ch}_contribution" in contributions_df.columns]
       
       # Ajouter la contribution de base
       if 'baseline_contribution' in contributions_df.columns:
           contrib_cols.append('baseline_contribution')
       
       # Calculer les contributions moyennes
       avg_contribs = {}
       for col in contrib_cols:
           channel = col.replace('_contribution', '')
           avg_contribs[channel] = contributions_df[col].mean()
       
       # Créer un DataFrame pour l'affichage
       contrib_df = pd.DataFrame({
           'Canal': list(avg_contribs.keys()),
           'Contribution moyenne (£)': list(avg_contribs.values()),
           'Contribution (%)': [v / contributions_df['predicted_revenue'].mean() * 100 for v in avg_contribs.values()]
       }).sort_values('Contribution moyenne (£)', ascending=False)
       
       # Afficher le tableau
       st.subheader("Contributions moyennes par canal")
       st.dataframe(contrib_df.style.format({
           'Contribution moyenne (£)': '{:.2f}',
           'Contribution (%)': '{:.2f}%'
       }))
       
       # Graphique de répartition des contributions
       st.subheader("Répartition des contributions")
       
       col1, col2 = st.columns(2)
       
       with col1:
           # Graphique à barres
           fig, ax = plt.subplots(figsize=(10, 6))
           bars = ax.bar(contrib_df['Canal'], contrib_df['Contribution moyenne (£)'])
           
           # Ajouter les valeurs sur les barres
           for bar in bars:
               height = bar.get_height()
               ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'£{height:.0f}', ha='center', va='bottom', fontsize=10)
           
           ax.set_xlabel('Canal')
           ax.set_ylabel('Contribution moyenne (£)')
           ax.set_title('Contribution moyenne par canal marketing')
           plt.xticks(rotation=45)
           plt.tight_layout()
           
           st.pyplot(fig)
       
       with col2:
           # Graphique en camembert
           fig, ax = plt.subplots(figsize=(10, 6))
           ax.pie(contrib_df['Contribution (%)'], labels=contrib_df['Canal'], autopct='%1.1f%%',
                  startangle=90, shadow=False)
           ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
           ax.set_title('Répartition des contributions (%)')
           plt.tight_layout()
           
           st.pyplot(fig)
       
       # Contributions dans le temps
       st.subheader("Évolution des contributions dans le temps")
       
       # Préparer les données
       contributions_df['date'] = pd.to_datetime(contributions_df['date'])
       contributions_df = contributions_df.sort_values('date')
       
       # Agréger par semaine pour une meilleure lisibilité
       contributions_df['week'] = contributions_df['date'].dt.isocalendar().week
       contributions_df['year'] = contributions_df['date'].dt.isocalendar().year
       
       weekly_contribs = contributions_df.groupby(['year', 'week']).agg({
           'date': 'first',
           **{col: 'mean' for col in contrib_cols}
       }).reset_index()
       
       # Tracer les contributions dans le temps
       fig, ax = plt.subplots(figsize=(12, 6))
       
       for col in contrib_cols:
           channel = col.replace('_contribution', '')
           ax.plot(weekly_contribs['date'], weekly_contribs[col], label=channel, linewidth=2)
       
       ax.set_xlabel('Date')
       ax.set_ylabel('Contribution (£)')
       ax.set_title('Évolution des contributions par canal')
       ax.legend(title='Canal', bbox_to_anchor=(1.05, 1), loc='upper left')
       ax.grid(True, alpha=0.3)
       plt.xticks(rotation=45)
       plt.tight_layout()
       
       st.pyplot(fig)
   else:
       st.info("Données de contribution non disponibles. Veuillez d'abord exécuter l'analyse MMM.")

elif page == "Allocation budgétaire":
   st.header("Optimisation de l'allocation budgétaire")
   
   if budget_df is not None and contributions_df is not None:
       # Préparer les données
       roi_data = {}
       for channel in config['marketing_channels']:
           roi_col = f"{channel}_roi"
           if roi_col in contributions_df.columns:
               roi_data[channel] = contributions_df[roi_col].median()
       
       # Créer un DataFrame pour l'affichage du ROI
       roi_df = pd.DataFrame({
           'Canal': list(roi_data.keys()),
           'ROI médian': list(roi_data.values())
       }).sort_values('ROI médian', ascending=False)
       
       # Afficher les ROI
       st.subheader("Retour sur investissement (ROI) par canal")
       st.dataframe(roi_df.style.format({
           'ROI médian': '{:.2f}'
       }))
       
       # Graphique du ROI
       fig, ax = plt.subplots(figsize=(10, 6))
       bars = ax.bar(roi_df['Canal'], roi_df['ROI médian'])
       
       # Ajouter les valeurs sur les barres
       for bar in bars:
           height = bar.get_height()
           ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=10)
       
       ax.set_xlabel('Canal')
       ax.set_ylabel('ROI (£ générés par £ dépensée)')
       ax.set_title('ROI médian par canal marketing')
       plt.xticks(rotation=45)
       plt.tight_layout()
       
       st.pyplot(fig)
       
       # Allocation budgétaire optimisée
       st.subheader("Allocation budgétaire optimisée")
       
       # Renommer les colonnes pour une meilleure présentation
       budget_display = budget_df.copy()
       if 'channel' in budget_display.columns:
           budget_display = budget_display.rename(columns={
               'channel': 'Canal',
               'budget': 'Budget (£)',
               'budget_pct': 'Budget (%)',
               'roi': 'ROI'
           })
           
           # Afficher le tableau
           st.dataframe(budget_display.style.format({
               'Budget (£)': '£{:.2f}',
               'Budget (%)': '{:.2f}%',
               'ROI': '{:.2f}'
           }))
           
           # Graphique d'allocation
           fig, ax1 = plt.subplots(figsize=(12, 6))
           
           x = budget_display['Canal']
           y1 = budget_display['Budget (£)']
           
           # Graphique en barres pour le budget
           bars = ax1.bar(x, y1, color='skyblue')
           
           # Ajouter les valeurs sur les barres
           for bar in bars:
               height = bar.get_height()
               ax1.text(bar.get_x() + bar.get_width()/2., height + 500,
                       f'£{height:.0f}', ha='center', va='bottom', fontsize=10)
           
           ax1.set_xlabel('Canal')
           ax1.set_ylabel('Budget alloué (£)', color='skyblue')
           ax1.tick_params(axis='y', labelcolor='skyblue')
           
           # Créer un deuxième axe Y pour le ROI
           ax2 = ax1.twinx()
           ax2.plot(x, budget_display['ROI'], 'ro-', linewidth=2)
           ax2.set_ylabel('ROI', color='red')
           ax2.tick_params(axis='y', labelcolor='red')
           
           plt.title('Allocation budgétaire optimisée et ROI par canal')
           plt.xticks(rotation=45)
           plt.tight_layout()
           
           st.pyplot(fig)
           
           # Comparaison avec le budget actuel
           st.subheader("Comparaison avec le budget actuel")
           st.write("""
           La comparaison avec le budget actuel n'est pas disponible car nous 
           utilisons des données marketing simulées. Dans un contexte réel, cette 
           section montrerait la comparaison entre l'allocation optimisée et l'allocation actuelle.
           """)
   else:
       st.info("Données d'allocation budgétaire non disponibles. Veuillez d'abord exécuter l'analyse MMM.")

elif page == "Simulateur":
   st.header("Simulateur d'allocation budgétaire")
   
   if budget_df is not None and contributions_df is not None:
       st.write("""
       Ce simulateur vous permet d'explorer différents scénarios d'allocation budgétaire
       et d'estimer leur impact sur les ventes.
       """)
       
       # Obtenir le budget total de base
       default_total_budget = budget_df['budget'].sum() if 'budget' in budget_df.columns else 100000
       
       # Input pour le budget total
       total_budget = st.slider(
           "Budget marketing total (£)",
           min_value=int(default_total_budget * 0.5),
           max_value=int(default_total_budget * 2),
           value=int(default_total_budget),
           step=5000
       )
       
       # Préparer les données pour le simulateur
       channels = config['marketing_channels']
       
       # Calculer les ROI médians par canal
       roi_data = {}
       for channel in channels:
           roi_col = f"{channel}_roi"
           if roi_col in contributions_df.columns:
               roi_data[channel] = contributions_df[roi_col].median()
       
       # Extraire les allocations par défaut
       default_allocations = {}
       if 'channel' in budget_df.columns and 'budget_pct' in budget_df.columns:
           for _, row in budget_df.iterrows():
               channel = row['channel']
               if channel in channels:
                   default_allocations[channel] = row['budget_pct']
       
       # Si pas d'allocations par défaut, répartir équitablement
       if not default_allocations:
           for channel in channels:
               default_allocations[channel] = 100 / len(channels)
       
       # Afficher les sliders pour ajuster l'allocation
       st.subheader("Ajuster l'allocation budgétaire (%)")
       
       allocations = {}
       col1, col2 = st.columns(2)
       
       for i, channel in enumerate(channels):
           default = default_allocations.get(channel, 100 / len(channels))
           with col1 if i < len(channels) / 2 else col2:
               allocations[channel] = st.slider(
                   f"{channel}",
                   min_value=0.0,
                   max_value=100.0,
                   value=float(default),
                   step=0.5,
                   format="%.1f%%"
               )
       
       # Normaliser les allocations pour qu'elles somment à 100%
       total_allocation = sum(allocations.values())
       if total_allocation != 0:
           for channel in allocations:
               allocations[channel] = (allocations[channel] / total_allocation) * 100
       
       # Calculer les budgets par canal
       budgets = {channel: (allocation / 100) * total_budget for channel, allocation in allocations.items()}
       
       # Calculer l'impact estimé sur les ventes
       estimated_contributions = {}
       baseline_contribution = contributions_df['baseline_contribution'].mean() if 'baseline_contribution' in contributions_df.columns else 0
       
       total_contribution = baseline_contribution
       for channel, budget in budgets.items():
           roi = roi_data.get(channel, 0)
           channel_contribution = budget * roi
           estimated_contributions[channel] = channel_contribution
           total_contribution += channel_contribution
       
       # Créer un DataFrame pour l'affichage des résultats
       results_df = pd.DataFrame({
           'Canal': list(budgets.keys()),
           'Budget (£)': list(budgets.values()),
           'Budget (%)': [allocation for allocation in allocations.values()],
           'ROI': [roi_data.get(channel, 0) for channel in budgets.keys()],
           'Contribution estimée (£)': [estimated_contributions[channel] for channel in budgets.keys()]
       })
       
       # Ajouter la ligne de base
       results_df = pd.concat([
           results_df,
           pd.DataFrame({
               'Canal': ['Baseline'],
               'Budget (£)': [0],
               'Budget (%)': [0],
               'ROI': [0],
               'Contribution estimée (£)': [baseline_contribution]
           })
       ]).reset_index(drop=True)
       
       # Afficher les résultats
       st.subheader("Résultats de la simulation")
       
       # Métriques principales
       col1, col2, col3 = st.columns(3)
       
       with col1:
           st.metric(
               label="Budget total",
               value=f"£{total_budget:,.2f}"
           )
       
       with col2:
           st.metric(
               label="Contribution marketing estimée",
               value=f"£{(total_contribution - baseline_contribution):,.2f}"
           )
       
       with col3:
           st.metric(
               label="Ventes totales estimées",
               value=f"£{total_contribution:,.2f}"
           )
       
       # Tableau détaillé
       st.dataframe(results_df.style.format({
           'Budget (£)': '£{:,.2f}',
           'Budget (%)': '{:.2f}%',
           'ROI': '{:.2f}',
           'Contribution estimée (£)': '£{:,.2f}'
       }))
       
       # Graphique des contributions estimées
       fig, ax = plt.subplots(figsize=(10, 6))
       
       # Exclure la baseline pour une meilleure visualisation
       plot_df = results_df[results_df['Canal'] != 'Baseline']
       
       bars = ax.bar(plot_df['Canal'], plot_df['Contribution estimée (£)'])
       
       # Ajouter les valeurs sur les barres
       for bar in bars:
           height = bar.get_height()
           ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                   f'£{height:,.0f}', ha='center', va='bottom', fontsize=10)
       
       ax.set_xlabel('Canal')
       ax.set_ylabel('Contribution estimée (£)')
       ax.set_title('Contribution estimée par canal marketing')
       plt.xticks(rotation=45)
       plt.tight_layout()
       
       st.pyplot(fig)
   else:
       st.info("Données requises non disponibles pour le simulateur. Veuillez d'abord exécuter l'analyse MMM.")

elif page == "À propos":
   st.header("À propos du projet")
   
   st.write("""
   ## Marketing Mix Modeling (MMM)
   
   Ce projet implémente une analyse de Marketing Mix Modeling (MMM) complète pour 
   optimiser l'allocation budgétaire marketing en utilisant les données Online Retail.
   
   ### Méthodologie
   
   Le MMM utilise des techniques statistiques avancées pour décomposer les ventes et 
   quantifier l'impact de chaque canal marketing. Notre approche comprend :
   
   1. **Prétraitement des données** : Nettoyage, agrégation et préparation des données
   2. **Modélisation des effets retardés** : Utilisation de transformations d'adstock pour 
      capturer les effets à long terme des campagnes marketing
   3. **Modélisation des effets de saturation** : Utilisation de transformations non linéaires 
      pour modéliser les rendements décroissants
   4. **Modélisation avancée** : Utilisation d'algorithmes comme LightGBM pour capturer 
      les relations complexes et non linéaires
   5. **Optimisation budgétaire** : Utilisation d'algorithmes d'optimisation pour maximiser le ROI
   
   ### Limites de l'analyse
   
   - Les dépenses marketing sont simulées car les données réelles ne sont pas disponibles
   - Certains facteurs externes peuvent ne pas être pris en compte
   - Les effets synergiques entre canaux sont difficiles à capturer parfaitement
   
   ### Références
   
   - [Facebook/Meta Robyn](https://github.com/facebookexperimental/Robyn)
   - [Google's MMM Python Library](https://github.com/google/lightweight_mmm)
   - Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017). Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects.
   """)
   
   st.subheader("Contact")
   st.write("""
   Pour toute question ou suggestion concernant ce projet, veuillez contacter :
   - **Nom** : Samir Elaissaouy
   - **Email** : elaissaouy.samir12@gmail.com
   - **GitHub** : [github.com/samirelais](https://github.com/samirelais)
   """)
