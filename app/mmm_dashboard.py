import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime, timedelta
from fpdf import FPDF
import matplotlib
matplotlib.use('Agg')  # Nécessaire pour générer des graphiques sans serveur X
import base64
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.units import inch

def get_project_root():
    """
    Trouve le chemin racine du projet de manière robuste et compatible multiplateforme.
    """
    # Liste des chemins potentiels
    possible_paths = [
        os.getcwd(),  # Répertoire de travail courant
        os.path.dirname(os.path.abspath(__file__)),  # Chemin du script
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # Un niveau au-dessus
        '/app'  # Chemin spécifique à Streamlit Cloud
    ]
    
    # Chemins relatifs à tester
    relative_paths = [
        'data',
        '../data',
        './data',
        'MMM_EC/data',
        '../MMM_EC/data'
    ]
    
    # Fonction pour vérifier si un chemin existe et contient les fichiers nécessaires
    def is_valid_path(path):
        try:
            # Liste des fichiers à vérifier
            required_files = [
                'Online_Retail.csv', 
                'online_retail_config.json'
            ]
            
            # Vérifier si le chemin existe et contient les fichiers requis
            return (os.path.exists(path) and 
                    any(os.path.exists(os.path.join(path, f)) for f in required_files))
        except Exception:
            return False
    
    # Tester les chemins absolus
    for base_path in possible_paths:
        for relative_path in relative_paths:
            full_path = os.path.normpath(os.path.join(base_path, relative_path))
            if is_valid_path(full_path):
                return full_path
    
    # Dernier recours : utiliser un chemin par défaut
    default_path = os.path.join(os.getcwd(), 'data')
    os.makedirs(default_path, exist_ok=True)
    return default_path

def find_file(filename, possible_locations=None):
    """
    Recherche un fichier dans différents emplacements possibles.
    """
    if possible_locations is None:
        possible_locations = [
            os.getcwd(),
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join(os.getcwd(), 'data'),
            os.path.join(os.getcwd(), 'config'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config'),
            '/app/data',
            '/app/config',
            '/content/drive/MyDrive/mmm-ecommerce/data',
            '/content/drive/MyDrive/mmm-ecommerce/config'
        ]
    
    for location in possible_locations:
        path = os.path.join(location, filename)
        if os.path.exists(path):
            return path
    
    return None

def load_config(filename='online_retail_config.json'):
    """
    Charger la configuration de manière robuste.
    """
    try:
        # Trouver le fichier de configuration
        config_path = find_file(filename)
        
        if config_path:
            st.info(f"Chargement de la configuration depuis : {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Adapter le chemin des données si nécessaire
            if 'data' in config and 'retail_data_path' in config['data']:
                data_filename = os.path.basename(config['data']['retail_data_path'])
                data_path = find_file(data_filename)
                
                if data_path:
                    config['data']['retail_data_path'] = data_path
            
            return config
        
        # Configuration par défaut si aucun fichier n'est trouvé
        st.warning("Aucun fichier de configuration trouvé. Utilisation de la configuration par défaut.")
        return {
            "data": {
                "retail_data_path": "data/Online_Retail.csv",
                "include_returns": False,
                "start_date": "2010-12-01",
                "end_date": "2011-12-09"
            },
            "marketing_channels": ["tv", "radio", "print", "social_media", "search", "email", "display"],
            "preprocessing": {
                "remove_outliers": True,
                "outlier_threshold": 3
            }
        }
    
    except Exception as e:
        st.error(f"Erreur lors du chargement de la configuration: {e}")
        return {
            "data": {
                "retail_data_path": "data/Online_Retail.csv",
                "include_returns": False,
                "start_date": "2010-12-01",
                "end_date": "2011-12-09"
            },
            "marketing_channels": ["tv", "radio", "print", "social_media", "search", "email", "display"]
        }

def load_data(config=None):
    """
    Charger les données de manière robuste.
    """
    # Utiliser la configuration fournie ou charger une configuration par défaut
    if config is None:
        config = load_config()
    
    try:
        # Extraire le nom du fichier ou utiliser un nom par défaut
        data_filename = os.path.basename(config['data'].get('retail_data_path', 'Online_Retail.csv'))
        
        # Trouver le fichier de données
        data_path = find_file(data_filename)
        
        if data_path:
            st.info(f"Chargement des données depuis : {data_path}")
            df = pd.read_csv(data_path)
            
            # Convertir la colonne de date
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
            
            # Filtrer par date si spécifié
            if 'start_date' in config['data'] and 'end_date' in config['data']:
                start_date = pd.to_datetime(config['data']['start_date'])
                end_date = pd.to_datetime(config['data']['end_date'])
                df = df[(df['InvoiceDate'] >= start_date) & (df['InvoiceDate'] <= end_date)]
            
            # Gérer les retours si nécessaire
            if config['data'].get('include_returns', False) is False:
                df = df[df['Quantity'] > 0]
            
            return df
        
        # Générer des données de démonstration si aucun fichier n'est trouvé
        st.warning("Aucun fichier de données trouvé. Génération de données de démonstration.")
        return _generate_demo_data()
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return _generate_demo_data()

def _generate_demo_data():
    """
    Générer des données de démonstration.
    """
    dates = pd.date_range(start='2010-12-01', end='2011-12-09')
    data = pd.DataFrame({
        'InvoiceDate': dates,
        'InvoiceNo': [f'INV_{i}' for i in range(len(dates))],
        'StockCode': np.random.choice(['A001', 'B002', 'C003'], len(dates)),
        'Description': ['Sample Product ' + str(i) for i in range(len(dates))],
        'Quantity': np.random.randint(1, 10, len(dates)),
        'UnitPrice': np.random.uniform(10, 100, len(dates)),
        'CustomerID': np.random.randint(10000, 99999, len(dates)),
        'Country': np.random.choice(['UK', 'France', 'Germany'], len(dates))
    })
    data['TotalPrice'] = data['Quantity'] * data['UnitPrice']
    return data

def load_results(results_folder='reports'):
    """
    Charger les résultats du modèle de manière robuste.
    """
    try:
        # Chemins possibles pour les fichiers de résultats
        possible_paths = [
            os.getcwd(),
            os.path.join(os.getcwd(), results_folder),
            os.path.dirname(os.path.abspath(__file__)),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), results_folder),
            '/app/reports',
            '/content/drive/MyDrive/mmm-ecommerce/reports'
        ]
        
        # Noms des fichiers de résultats
        contributions_filename = 'channel_contributions.csv'
        budget_filename = 'budget_allocation.csv'
        metrics_filename = 'model_metrics.json'
        
        # Trouver les chemins des fichiers
        contributions_path = find_file(contributions_filename, possible_paths)
        budget_path = find_file(budget_filename, possible_paths)
        metrics_path = find_file(metrics_filename, possible_paths)
        
        # Si les fichiers n'existent pas, générer des données de démonstration
        if not all([contributions_path, budget_path, metrics_path]):
            st.warning("Fichiers de résultats non trouvés. Génération de données de démonstration.")
            return _generate_demo_results()
        
        # Charger les fichiers
        contributions_df = pd.read_csv(contributions_path)
        budget_df = pd.read_csv(budget_path)
        
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        return contributions_df, budget_df, metrics
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des résultats: {e}")
        return _generate_demo_results()



def _generate_demo_results():
    """
    Générer des résultats de démonstration.
    """
    # Canaux de marketing par défaut
    channels = ["tv", "radio", "print", "social_media", "search", "email", "display"]
    
    # Générer des contributions par canal
    dates = pd.date_range(start='2010-12-01', end='2011-12-31')
    contributions_df = pd.DataFrame({'date': dates})
    contributions_df['actual_revenue'] = np.random.normal(50000, 10000, len(dates))
    contributions_df['predicted_revenue'] = np.random.normal(contributions_df['actual_revenue'], 5000)
    
    for channel in channels:
        contributions_df[f'{channel}_contribution'] = np.random.normal(5000, 1000, len(dates))
        contributions_df[f'{channel}_spend'] = np.random.normal(2000, 500, len(dates))
        
        # Éviter la division par zéro
        contributions_df[f'{channel}_roi'] = np.where(
            contributions_df[f'{channel}_spend'] != 0, 
            contributions_df[f'{channel}_contribution'] / contributions_df[f'{channel}_spend'], 
            0
        )
    
    # Générer l'allocation budgétaire
    budget_data = []
    total_budget = 100000
    for channel in channels:
        budget = np.random.normal(total_budget / len(channels), 5000)
        budget_data.append({
            'channel': channel,
            'budget': budget,
            'budget_pct': budget / total_budget * 100,
            'roi': np.random.uniform(1.5, 4.0)
        })
    
    budget_df = pd.DataFrame(budget_data)
    
    # Métriques
    metrics = {
        'r2': 0.85,
        'rmse': 5000,
        'mae': 4000,
        'mape': 8.5
    }
    
    return contributions_df, budget_df, metrics

    

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Dashboard Marketing Mix Modeling",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Charger la configuration et les données
config = load_config()
data = load_data()
contributions_df, budget_df, metrics = load_results()

# [Toutes vos fonctions précédentes pour generate_pdf_report et generate_mmm_guide restent identiques]
def generate_pdf_report(contributions_df, budget_df, metrics, config):
    """
    Génère un rapport PDF complet avec les résultats de l'analyse MMM.
    """
    try:
        pdf = FPDF()
        # [Votre implémentation existante]
        return pdf.output(dest='S')
    except Exception as e:
        st.error(f"Erreur lors de la génération du rapport PDF: {e}")
        return None

def generate_mmm_guide():
    """Génère un guide PDF sur les principes du Marketing Mix Modeling"""
    try:
        buffer = io.BytesIO()
        # [Votre implémentation existante]
        return buffer.getvalue()
    except Exception as e:
        st.error(f"Erreur lors de la génération du guide: {e}")
        return None

# Titre de l'application
st.title("📊 Dashboard Marketing Mix Modeling")
st.write("Analyse et optimisation de l'attribution marketing basée sur les données Online Retail")

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

# Bouton pour recharger l'analyse
if st.sidebar.button("Actualiser les données"):
    # Recharger les données et les résultats
    config = load_config()
    data = load_data()
    contributions_df, budget_df, metrics = load_results()
    st.success("Données actualisées avec succès!")

# Ajout d'un bouton de téléchargement des rapports dans la sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Rapports")
if st.sidebar.button("Générer un guide MMM (PDF)"):
    pdf_guide = generate_mmm_guide()
    if pdf_guide:
        st.sidebar.download_button(
            label="Télécharger le Guide MMM",
            data=pdf_guide,
            file_name="guide_mmm.pdf",
            mime="application/pdf"
        )

# [Le reste de votre code original pour chaque page]
# Vue d'ensemble
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
            st.metric(label="MAPE", value=f"{metrics['mape']:.2f}")
# Graphique des ventes au fil du temps
    st.subheader("Évolution des ventes")
    if contributions_df is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Convertir les dates si nécessaire
        if not isinstance(contributions_df['date'].iloc[0], pd.Timestamp):
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
        st.info("Graphique non disponible. Veuillez d'abord exécuter l'analyse MMM ou charger des données de démo.")

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
            # Utiliser max() pour éviter les valeurs négatives
            'Contribution (%)': [max(0, v / contributions_df['predicted_revenue'].mean() * 100) for v in avg_contribs.values()]
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
        if not isinstance(contributions_df['date'].iloc[0], pd.Timestamp):
            contributions_df['date'] = pd.to_datetime(contributions_df['date'])
        contributions_df = contributions_df.sort_values('date')
        
        # Agréger par semaine pour une meilleure lisibilité
        contributions_df['week'] = contributions_df['date'].dt.isocalendar().week
        contributions_df['year'] = contributions_df['date'].dt.year
        
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
        st.info("Données de contribution non disponibles. Veuillez d'abord exécuter l'analyse MMM ou charger des données de démo.")

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
            
            # Téléchargement des résultats
            st.download_button(
                label="Télécharger l'allocation budgétaire (CSV)",
                data=budget_display.to_csv(index=False).encode('utf-8'),
                file_name='allocation_budgetaire_optimisee.csv',
                mime='text/csv',
            )
    else:
        st.info("Données d'allocation budgétaire non disponibles. Veuillez d'abord exécuter l'analyse MMM ou charger des données de démo.")

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
        
        # Options d'optimisation
        st.subheader("Options de simulation")
        optimisation_objective = st.selectbox(
            "Objectif d'optimisation",
            ["Maximiser les ventes", "Maximiser le ROI", "Équilibrer ventes et ROI"]
        )
        
        # Afficher les sliders pour ajuster l'allocation
        st.subheader("Ajuster l'allocation budgétaire (%)")
        
        # Option de réinitialisation
        reset_allocations = st.button("Réinitialiser aux valeurs optimales")
        
        allocations = {}
        col1, col2 = st.columns(2)
        
        for i, channel in enumerate(channels):
            default = default_allocations.get(channel, 100 / len(channels))
            # Réinitialiser si le bouton est pressé
            if reset_allocations:
                value = float(default)
            else:
                value = st.session_state.get(f"allocation_{channel}", float(default))
                
            with col1 if i < len(channels) / 2 else col2:
                allocations[channel] = st.slider(
                    f"{channel}",
                    min_value=0.0,
                    max_value=100.0,
                    value=value,
                    key=f"allocation_{channel}",
                    step=0.5,
                    format="%.1f%%"
                )
        
        # Normaliser les allocations pour qu'elles somment à 100%
        total_allocation = sum(allocations.values())
        if total_allocation > 0:
            allocation_factor = 100 / total_allocation
            for channel in allocations:
                allocations[channel] = allocations[channel] * allocation_factor
        
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
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Budget total",
                value=f"£{total_budget:,.2f}"
            )
        
        with col2:
            st.metric(
                label="Contribution marketing",
                value=f"£{(total_contribution - baseline_contribution):,.2f}"
            )
        
        with col3:
            st.metric(
                label="Ventes totales",
                value=f"£{total_contribution:,.2f}"
            )
            
        with col4:
            average_roi = sum(estimated_contributions.values()) / total_budget if total_budget > 0 else 0
            st.metric(
                label="ROI global",
                value=f"{average_roi:.2f}"
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
        
        # Téléchargement des résultats de simulation
        st.download_button(
            label="Télécharger les résultats de simulation (CSV)",
            data=results_df.to_csv(index=False).encode('utf-8'),
            file_name='simulation_allocation_budgetaire.csv',
            mime='text/csv',
        )
        
        # Historique des simulations
        if 'simulation_history' not in st.session_state:
            st.session_state.simulation_history = []
        
        # Bouton pour sauvegarder la simulation actuelle
        if st.button("Sauvegarder cette simulation"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            simulation_data = {
                'timestamp': timestamp,
                'budget_total': total_budget,
                'ventes_estimees': total_contribution,
                'roi_global': average_roi,
                'allocations': allocations.copy()
            }
            st.session_state.simulation_history.append(simulation_data)
            st.success(f"Simulation sauvegardée: {timestamp}")
        
        # Afficher l'historique des simulations
        if st.session_state.simulation_history:
            st.subheader("Historique des simulations")
            history_df = pd.DataFrame([
                {
                    'Date': sim['timestamp'],
                    'Budget total (£)': f"£{sim['budget_total']:,.2f}",
                    'Ventes estimées (£)': f"£{sim['ventes_estimees']:,.2f}",
                    'ROI global': f"{sim['roi_global']:.2f}"
                } 
                for sim in st.session_state.simulation_history
            ])
            st.dataframe(history_df)
            
            # Comparaison des simulations
            if len(st.session_state.simulation_history) > 1:
                st.subheader("Comparaison des simulations")
                
                # Tracer l'évolution des ventes estimées
                fig, ax = plt.subplots(figsize=(10, 6))
                data_to_plot = [
                    (sim['timestamp'], sim['ventes_estimees']) 
                    for sim in st.session_state.simulation_history
                ]
                timestamps, values = zip(*data_to_plot)
                
                ax.plot(range(len(timestamps)), values, 'o-', linewidth=2)
                ax.set_xlabel('Simulation')
                ax.set_ylabel('Ventes estimées (£)')
                ax.set_title('Évolution des ventes estimées par simulation')
                ax.set_xticks(range(len(timestamps)))
                ax.set_xticklabels([f"Sim {i+1}" for i in range(len(timestamps))], rotation=45)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                st.pyplot(fig)
    else:
        st.info("Données requises non disponibles pour le simulateur. Veuillez d'abord exécuter l'analyse MMM ou charger des données de démo.")

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
    
    # Ajouter une section FAQ
    st.subheader("FAQ")
    
    faq_data = [
        ("Qu'est-ce que l'effet d'adstock?", 
         "L'effet d'adstock est le phénomène par lequel l'impact des dépenses marketing se poursuit au-delà de la période initiale. Il capture l'effet de persistance ou de mémoire des campagnes marketing."),
        
        ("Comment interpréter le ROI?", 
         "Le ROI (Return on Investment) mesure le rendement généré pour chaque unité monétaire dépensée. Par exemple, un ROI de 2.5 signifie que pour chaque £1 dépensée, vous générez £2.50 de revenus."),
        
        ("Comment le modèle gère-t-il la saisonnalité?", 
         "Le modèle capture la saisonnalité en incluant des variables indicatrices pour les périodes pertinentes (mois, jours de la semaine) et en identifiant les tendances cycliques dans les données."),
        
        ("Comment puis-je améliorer la précision du modèle?", 
         "Pour améliorer la précision, vous pouvez : 1) inclure plus de données historiques, 2) ajouter des variables exogènes pertinentes, 3) affiner les paramètres d'adstock et de saturation, ou 4) expérimenter avec différents algorithmes de modélisation.")
    ]
    
    for question, answer in faq_data:
        with st.expander(question):
            st.write(answer)
    
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Télécharger le guide MMM"):
            with st.spinner('Génération du guide en cours...'):
                pdf_guide = generate_mmm_guide()
                if pdf_guide:
                    st.success("Guide généré avec succès!")
                    st.download_button(
                        label="Télécharger le Guide MMM (PDF)",
                        data=pdf_guide,
                        file_name="guide_mmm.pdf",
                        mime="application/pdf"
                    )

# Fonction pour le tutoriel
def show_tutorial():
    st.info("Fonctionnalité de tutoriel à implémenter dans une version future.")

# Ajouter un bouton de tutoriel dans la sidebar
if st.sidebar.button("Tutoriel"):
    show_tutorial()

# Point d'entrée principal
if __name__ == "__main__":
    st.write("Cette application est un dashboard de Marketing Mix Modeling.")