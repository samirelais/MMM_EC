import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
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

# Définition des chemins
# Utiliser des chemins relatifs au lieu de chemins absolus
base_path = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir(base_path):
    base_path = os.getcwd()

# Pour remonter d'un niveau depuis le dossier app/
parent_path = os.path.dirname(base_path)

# Fonction pour charger la configuration
@st.cache_data
def load_config():
    try:
        config_path = os.path.join(parent_path, "config/online_retail_config.json")
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError as e:
        st.warning(f"Fichier de configuration non trouvé: {e}. Chemins vérifiés: {config_path}")
        # Configuration par défaut
        return {
            "marketing_channels": ["search", "social", "email", "display", "affiliates"],
            "data_path": "data/online_retail.csv",
            "date_column": "InvoiceDate",
            "target_column": "Revenue",
            "time_granularity": "daily"
        }






# Fonction pour charger la configuration
@st.cache_data
def load_config():
    try:
        config_path = os.path.join(base_path, "config/online_retail_config.json")
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError as e:
        st.warning(f"Fichier de configuration non trouvé: {e}. Chemins vérifiés: {config_path}")
        # Configuration par défaut
        return {
            "marketing_channels": ["search", "social", "email", "display", "affiliates"],
            "data_path": "data/online_retail.csv",
            "date_column": "InvoiceDate",
            "target_column": "Revenue",
            "time_granularity": "daily"
        }

# Fonction pour générer un rapport PDF
def generate_pdf_report(contributions_df, budget_df, metrics, config):
    """
    Génère un rapport PDF complet avec les résultats de l'analyse MMM.
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Paramètres de police
        pdf.set_font('Arial', 'B', 16)
        
        # Titre du rapport
        pdf.cell(190, 10, 'Rapport Marketing Mix Modeling', 0, 1, 'C')
        pdf.ln(10)
        
        # Sous-titre et date
        pdf.set_font('Arial', 'I', 10)
        current_date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        pdf.cell(190, 5, f'Généré le: {current_date}', 0, 1, 'R')
        pdf.ln(5)
        
        # Résumé des performances du modèle
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 10, '1. Performance du Modèle', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        pdf.cell(190, 7, f"R²: {metrics['r2']:.3f}   RMSE: {metrics['rmse']:.2f}   MAE: {metrics['mae']:.2f}   MAPE: {metrics['mape']:.2f}%", 0, 1, 'L')
        pdf.ln(5)
        
        # Contributions par canal
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 10, '2. Contributions par Canal', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        
        # Préparer les données pour le tableau
        channels = config['marketing_channels']
        contrib_cols = [f"{ch}_contribution" for ch in channels if f"{ch}_contribution" in contributions_df.columns]
        
        if 'baseline_contribution' in contributions_df.columns:
            contrib_cols.append('baseline_contribution')
            
        avg_contribs = {}
        for col in contrib_cols:
            channel = col.replace('_contribution', '')
            avg_contribs[channel] = contributions_df[col].mean()
            
        # Entêtes du tableau
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(60, 7, 'Canal', 1, 0, 'C', 1)
        pdf.cell(60, 7, 'Contribution moyenne (£)', 1, 0, 'C', 1)
        pdf.cell(60, 7, 'Contribution (%)', 1, 1, 'C', 1)
        
        # Données du tableau
        total_revenue = contributions_df['predicted_revenue'].mean()
        pdf.set_fill_color(255, 255, 255)
        
        for channel, value in avg_contribs.items():
            contribution_pct = max(0, value / total_revenue * 100)
            pdf.cell(60, 7, channel, 1, 0, 'L')
            pdf.cell(60, 7, f'{value:.2f}', 1, 0, 'R')
            pdf.cell(60, 7, f'{contribution_pct:.2f}%', 1, 1, 'R')
        
        pdf.ln(5)
        
        # Allocation budgétaire
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 10, '3. Allocation Budgétaire Optimisée', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        
        # Entêtes du tableau
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(47, 7, 'Canal', 1, 0, 'C', 1)
        pdf.cell(47, 7, 'Budget (£)', 1, 0, 'C', 1)
        pdf.cell(47, 7, 'Budget (%)', 1, 0, 'C', 1)
        pdf.cell(47, 7, 'ROI', 1, 1, 'C', 1)
        
        # Données du tableau de budget
        if 'channel' in budget_df.columns:
            for _, row in budget_df.iterrows():
                pdf.cell(47, 7, row['channel'], 1, 0, 'L')
                pdf.cell(47, 7, f'{row["budget"]:.2f}', 1, 0, 'R')
                pdf.cell(47, 7, f'{row["budget_pct"]:.2f}%', 1, 0, 'R')
                pdf.cell(47, 7, f'{row["roi"]:.2f}', 1, 1, 'R')
        
        pdf.ln(5)
        
        # Ajouter un graphique d'évolution des ventes
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 10, '4. Évolution des Ventes', 0, 1, 'L')
        
        # Créer le graphique
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # S'assurer que les dates sont au bon format
        if not isinstance(contributions_df['date'].iloc[0], pd.Timestamp):
            contributions_df['date'] = pd.to_datetime(contributions_df['date'])
        
        # Trier par date
        contributions_df = contributions_df.sort_values('date')
        
        # Tracer les ventes réelles et prédites
        ax.plot(contributions_df['date'], contributions_df['actual_revenue'], 
                label='Ventes réelles', color='blue')
        ax.plot(contributions_df['date'], contributions_df['predicted_revenue'], 
                label='Ventes prédites', color='red', linestyle='--')
        
        # Formatage du graphique
        ax.set_xlabel('Date')
        ax.set_ylabel('Ventes (£)')
        ax.set_title('Évolution des ventes - Réelles vs Prédites')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Sauvegarder le graphique en mémoire
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        
        # Ajouter le graphique au PDF
        img_width = 190
        img_height = 90
        x_position = 10
        y_position = pdf.get_y()
        
        pdf.image(img_buffer, x=x_position, y=y_position, w=img_width, h=img_height)
        
        # Ajuster la position Y après l'image
        pdf.set_y(y_position + img_height + 10)
        
        # Conclusion et recommandations
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(190, 10, '5. Recommandations', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        
        # Chercher le canal avec le meilleur ROI
        best_roi_channel = 'N/A'
        if 'channel' in budget_df.columns:
            best_roi_row = budget_df.loc[budget_df['roi'].idxmax()]
            best_roi_channel = best_roi_row['channel']
        
        # Ajouter des recommandations basées sur l'analyse
        pdf.multi_cell(190, 7, 
            f"Basé sur l'analyse MMM, voici nos recommandations clés:\n\n"
            f"1. Le canal '{best_roi_channel}' présente le meilleur ROI et devrait être privilégié.\n"
            f"2. L'allocation budgétaire optimisée présentée dans ce rapport permettrait d'améliorer le retour sur investissement global.\n"
            f"3. Une révision trimestrielle de l'allocation est recommandée pour s'adapter aux évolutions du marché.\n"
            f"4. Des tests A/B devraient être conduits pour valider empiriquement l'efficacité des différents canaux."
        )
        
        return pdf.output(dest='S')
    except Exception as e:
        st.error(f"Erreur lors de la génération du rapport PDF: {e}")
        return None

def generate_mmm_guide():
    """Génère un guide PDF sur les principes du Marketing Mix Modeling"""
    try:
        # Créer un buffer pour stocker le PDF
        buffer = io.BytesIO()
        
        # Créer le document PDF
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = styles['Heading1']
        subtitle_style = styles['Heading2']
        subsubtitle_style = styles['Heading3']
        normal_style = styles['Normal']
        
        # Style personnalisé pour les paragraphes du guide
        guide_style = ParagraphStyle(
            'GuideStyle',
            parent=normal_style,
            leading=14,  # Espacement entre les lignes
            spaceAfter=12  # Espace après chaque paragraphe
        )
        
        # Titre principal
        elements.append(Paragraph("Guide du Marketing Mix Modeling", title_style))
        elements.append(Spacer(1, 0.3*inch))
        
        # Introduction
        elements.append(Paragraph("Introduction", subtitle_style))
        intro_text = """
        Le Marketing Mix Modeling (MMM) est une technique statistique utilisée pour quantifier l'impact des différentes 
        activités marketing sur les ventes. Ce guide vous présente les concepts fondamentaux du MMM, sa méthodologie
        et son application dans un contexte d'entreprise.
        """
        elements.append(Paragraph(intro_text, guide_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Principes fondamentaux
        elements.append(Paragraph("Principes fondamentaux", subtitle_style))
        
        elements.append(Paragraph("Qu'est-ce que le MMM?", subsubtitle_style))
        mmm_text = """
        Le Marketing Mix Modeling est une approche analytique qui utilise des techniques de régression statistique pour 
        évaluer l'efficacité des différents canaux marketing et quantifier leur impact sur les ventes ou d'autres 
        indicateurs de performance. L'objectif principal est de déterminer le retour sur investissement (ROI) de chaque 
        canal et d'optimiser l'allocation des ressources marketing.
        """
        elements.append(Paragraph(mmm_text, guide_style))
        
        elements.append(Paragraph("Variables clés du MMM", subsubtitle_style))
        var_text = """
        Un modèle MMM prend généralement en compte quatre types de variables:
        • Variables dépendantes: Ventes, revenus ou autres KPIs à expliquer
        • Variables marketing: Dépenses publicitaires, GRP, impressions par canal
        • Variables de contrôle: Prix, distribution, saisonnalité, concurrence
        • Variables externes: Facteurs macroéconomiques, météo, événements spéciaux
        """
        elements.append(Paragraph(var_text, guide_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Méthodologie
        elements.append(Paragraph("Méthodologie", subtitle_style))
        
        elements.append(Paragraph("1. Collecte et préparation des données", subsubtitle_style))
        data_text = """
        La première étape consiste à rassembler toutes les données pertinentes: historique des ventes, 
        dépenses marketing par canal, prix, promotions, et variables externes. Les données doivent être 
        nettoyées, agrégées au même niveau temporel (généralement hebdomadaire ou mensuel) et explorées
        pour détecter des tendances ou anomalies.
        """
        elements.append(Paragraph(data_text, guide_style))
        
        elements.append(Paragraph("2. Modélisation des effets marketing", subsubtitle_style))
        model_text = """
        Un modèle MMM complet prend en compte trois phénomènes essentiels:
        • Effet Adstock (retardé): Les effets du marketing persistent au-delà de la période initiale
        • Effet de saturation: Rendements décroissants à mesure que les dépenses augmentent
        • Effet synergique: Interactions entre différents canaux marketing
        
        Ces effets sont modélisés à l'aide de transformations mathématiques comme les fonctions d'Adstock, 
        les fonctions Hill, Michaelis-Menten ou logarithmiques pour les effets de saturation.
        """
        elements.append(Paragraph(model_text, guide_style))
        
        elements.append(Paragraph("3. Construction et validation du modèle", subsubtitle_style))
        validation_text = """
        Les modèles MMM peuvent être construits avec différentes techniques, allant de la régression linéaire 
        aux algorithmes plus avancés comme le gradient boosting (XGBoost, LightGBM) ou les modèles bayésiens.
        
        La validation du modèle est cruciale et implique:
        • Validation croisée pour éviter le surajustement
        • Tests de robustesse avec différentes périodes
        • Vérification des hypothèses statistiques
        • Comparaison des résultats avec les données historiques
        """
        elements.append(Paragraph(validation_text, guide_style))
        
        elements.append(Paragraph("4. Analyse des résultats et optimisation", subsubtitle_style))
        results_text = """
        L'analyse des résultats permet d'identifier:
        • La contribution de chaque canal aux ventes totales
        • Le ROI par canal (retour généré pour chaque euro investi)
        • Le point de saturation pour chaque canal
        
        Ces informations servent ensuite à optimiser l'allocation budgétaire future, généralement en utilisant 
        des algorithmes d'optimisation sous contraintes.
        """
        elements.append(Paragraph(results_text, guide_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Applications et limites
        elements.append(Paragraph("Applications et limites", subtitle_style))
        
        elements.append(Paragraph("Applications pratiques", subsubtitle_style))
        applications_text = """
        Le MMM est utilisé pour:
        • Optimiser l'allocation budgétaire entre canaux
        • Planifier les futures campagnes marketing
        • Justifier les investissements marketing auprès de la direction
        • Comprendre l'efficacité relative des différents canaux
        • Simuler différents scénarios budgétaires
        """
        elements.append(Paragraph(applications_text, guide_style))
        
        elements.append(Paragraph("Limites et défis", subsubtitle_style))
        limits_text = """
        Le MMM présente certaines limites:
        • Nécessite d'importantes quantités de données historiques (2-3 ans minimum)
        • Difficulté à capturer les effets à très long terme (brand building)
        • Complexité à modéliser les interactions entre canaux
        • Sensibilité aux changements structurels du marché
        • Incapacité à mesurer les effets au niveau individuel
        """
        elements.append(Paragraph(limits_text, guide_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Tendances récentes
        elements.append(Paragraph("Tendances récentes", subtitle_style))
        trends_text = """
        Le MMM continue d'évoluer avec:
        • L'intégration de l'apprentissage automatique pour des modèles plus précis
        • L'unification avec les modèles d'attribution digitale (Unified MMM)
        • L'utilisation de données granulaires au niveau géographique ou démographique
        • L'incorporation de modèles bayésiens pour une meilleure quantification de l'incertitude
        • L'automatisation du processus avec des plateformes comme Meta Robyn ou Google LightweightMMM
        """
        elements.append(Paragraph(trends_text, guide_style))
        
        # Conclusion
        elements.append(Paragraph("Conclusion", subtitle_style))
        conclusion_text = """
        Le Marketing Mix Modeling reste un outil essentiel dans l'arsenal analytique des entreprises, 
        permettant une approche basée sur les données pour optimiser l'efficacité marketing. Bien que 
        présentant certaines limites, il offre une vision holistique de l'impact marketing difficile 
        à obtenir par d'autres méthodes. Son évolution continue avec l'incorporation de nouvelles 
        techniques et l'intégration de données plus détaillées en fait un domaine d'innovation constante.
        """
        elements.append(Paragraph(conclusion_text, guide_style))
        
        # Générer le PDF
        doc.build(elements)
        return buffer.getvalue()
        
    except Exception as e:
        st.error(f"Erreur lors de la génération du guide: {e}")
        return None
# Fonction pour charger les résultats du modèle
@st.cache_data
def load_results():
    try:
        # Chemins des fichiers
        contributions_path = os.path.join(base_path, "reports/channel_contributions.csv")
        budget_path = os.path.join(base_path, "reports/budget_allocation.csv")
        metrics_path = os.path.join(base_path, "reports/model_metrics.json")
        
        # Vérifier si les fichiers existent
        for path in [contributions_path, budget_path, metrics_path]:
            if not os.path.exists(path):
                st.warning(f"Fichier non trouvé: {path}")
        
        # Charger les contributions
        contributions_df = pd.read_csv(contributions_path)
        
        # Charger l'allocation budgétaire
        budget_df = pd.read_csv(budget_path)
        
        # Charger les métriques
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        return contributions_df, budget_df, metrics
    except FileNotFoundError as e:
        st.warning(f"Résultats du modèle non trouvés: {e}")
        return None, None, None
    except Exception as e:
        st.error(f"Erreur lors du chargement des résultats: {e}")
        return None, None, None

# Fonction pour exécuter l'analyse MMM (si les modules sont importés)
def run_mmm_analysis():
    try:
        st.info("Exécution de l'analyse MMM en cours...")
        
        # Vérifier si les modules sont disponibles
        if 'OnlineRetailLoader' not in globals():
            st.error("Les modules nécessaires ne sont pas disponibles. Assurez-vous d'être sur Google Colab avec les modules importés.")
            return False
        
        # Charger les données
        config = load_config()
        loader = OnlineRetailLoader(os.path.join(base_path, config["data_path"]))
        data = loader.load_and_prepare_data()
        
        # Créer et entraîner le modèle MMM
        model = MMMModel(config)
        model_results = model.train(data)
        
        # Générer les rapports
        model.generate_reports(model_results)
        
        st.success("Analyse MMM terminée avec succès!")
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'exécution de l'analyse MMM: {e}")
        return False

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

# Sidebar pour exécuter l'analyse MMM ou charger des données de démo
if st.sidebar.button("Exécuter l'analyse MMM"):
    run_mmm_analysis()

# Tenter de charger les résultats
try:
    contributions_df, budget_df, metrics = load_results()
    if all(x is None for x in [contributions_df, budget_df, metrics]):
        st.warning("Aucun résultat de modèle disponible. Utilisez 'Exécuter l'analyse MMM' pour générer des résultats.")
except Exception as e:
    st.error(f"Erreur lors du chargement des résultats: {e}")
    contributions_df, budget_df, metrics = None, None, None

# Vérifier si les résultats sont disponibles
if contributions_df is None:
    # Option de démo avec données simulées si les résultats ne sont pas disponibles
    if st.sidebar.button("Charger des données de démo"):
        # Générer des données de démo pour les contributions
        dates = pd.date_range(start='2010-12-01', end='2011-12-31')
        channels = config['marketing_channels'] + ['baseline']
        
        # Initialiser le DataFrame
        contributions_df = pd.DataFrame({'date': dates})
        contributions_df['actual_revenue'] = np.random.normal(50000, 10000, len(dates))
        contributions_df['predicted_revenue'] = np.random.normal(contributions_df['actual_revenue'], 5000)
        
        # Ajouter les contributions par canal
        for channel in channels:
            if channel == 'baseline':
                contributions_df['baseline_contribution'] = np.random.normal(30000, 5000, len(dates))
            else:
                contributions_df[f'{channel}_contribution'] = np.random.normal(5000, 1000, len(dates))
                contributions_df[f'{channel}_spend'] = np.random.normal(2000, 500, len(dates))
                contributions_df[f'{channel}_roi'] = contributions_df[f'{channel}_contribution'] / contributions_df[f'{channel}_spend']
        
        # Générer des données de démo pour l'allocation budgétaire
        budget_data = []
        for channel in config['marketing_channels']:
            budget_data.append({
                'channel': channel,
                'budget': np.random.normal(10000, 2000),
                'budget_pct': 0,  # Sera calculé ci-dessous
                'roi': np.random.uniform(1.5, 4.0)
            })
        
        budget_df = pd.DataFrame(budget_data)
        total_budget = budget_df['budget'].sum()
        budget_df['budget_pct'] = budget_df['budget'] / total_budget * 100
        
        # Générer des métriques de démo
        metrics = {
            'r2': 0.85,
            'rmse': 5000,
            'mae': 4000,
            'mape': 8.5
        }
        
        st.success("Données de démo chargées avec succès!")

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
    tutorial_steps = [
        {
            "title": "Bienvenue sur le Dashboard MMM!",
            "text": "Ce tutoriel rapide vous guidera à travers les fonctionnalités principales de l'application.",
            "element": "header",
        },
        {
            "title": "Navigation",
            "text": "Utilisez le menu de navigation pour explorer les différentes sections du dashboard.",
            "element": ".stSelectbox",
        },
        {
            "title": "Exécuter l'analyse",
            "text": "Cliquez sur ce bouton pour exécuter ou mettre à jour l'analyse MMM.",
            "element": "button[data-baseweb='button']:contains('Exécuter l'analyse MMM')",
        },
        {
            "title": "Simulateur",
            "text": "Explorez différents scénarios d'allocation budgétaire pour maximiser vos ventes.",
            "element": "div:contains('Simulateur d'allocation budgétaire')",
        },
        {
            "title": "Téléchargement",
            "text": "Téléchargez les résultats pour les utiliser dans d'autres outils comme Excel.",
            "element": "button:contains('Télécharger')",
        },
        {
            "title": "Besoin d'aide?",
            "text": "Consultez la section FAQ dans la page 'À propos' pour des réponses aux questions courantes.",
            "element": "div:contains('FAQ')",
        },
    ]

    # Cette fonction est un placeholder - la mise en œuvre effective d'un tutoriel 
    # nécessiterait une bibliothèque JavaScript comme Intro.js intégrée à Streamlit
    st.info("Fonctionnalité de tutoriel à implémenter dans une version future.")

# Ajouter un bouton de tutoriel dans la sidebar
if st.sidebar.button("Tutoriel"):
    show_tutorial()

# Ajouter une option pour télécharger un rapport complet
st.sidebar.markdown("---")
st.sidebar.subheader("Rapports")
if st.sidebar.button("Générer un rapport complet"):
    if contributions_df is not None and budget_df is not None and metrics is not None:
        try:
            # Créer un buffer pour stocker le PDF
            buffer = io.BytesIO()
            
            # Créer le document PDF
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            elements = []
            
            # Styles
            styles = getSampleStyleSheet()
            title_style = styles['Heading1']
            subtitle_style = styles['Heading2']
            normal_style = styles['Normal']
            
            # Titre principal
            elements.append(Paragraph("Rapport Marketing Mix Modeling", title_style))
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph(f"Date: {datetime.now().strftime('%d/%m/%Y')}", normal_style))
            elements.append(Spacer(1, 0.5*inch))
            
            # Métriques du modèle
            elements.append(Paragraph("Performance du modèle", subtitle_style))
            
            # Créer un tableau pour les métriques
            metrics_data = [["Métrique", "Valeur"],
                           ["R²", f"{metrics['r2']:.3f}"],
                           ["RMSE", f"{metrics['rmse']:.2f}"],
                           ["MAE", f"{metrics['mae']:.2f}"],
                           ["MAPE", f"{metrics['mape']:.2f}%"]]
            
            metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(metrics_table)
            elements.append(Spacer(1, 0.3*inch))
            
            # Contributions par canal
            elements.append(Paragraph("Contributions par canal", subtitle_style))
            
            # Préparer les données de contribution
            channels = config['marketing_channels']
            contrib_cols = [f"{ch}_contribution" for ch in channels if f"{ch}_contribution" in contributions_df.columns]
            if 'baseline_contribution' in contributions_df.columns:
                contrib_cols.append('baseline_contribution')
                
            # Calculer les contributions moyennes
            avg_contribs = {}
            for col in contrib_cols:
                channel = col.replace('_contribution', '')
                avg_contribs[channel] = contributions_df[col].mean()
            
            total_revenue = contributions_df['predicted_revenue'].mean()
                
            # Créer un tableau pour les contributions
            contrib_data = [["Canal", "Contribution (£)", "Pourcentage (%)"]]
            for channel, value in avg_contribs.items():
                pct = max(0, value / total_revenue * 100)
                contrib_data.append([channel, f"{value:.2f}", f"{pct:.2f}%"])
            
            contrib_table = Table(contrib_data, colWidths=[2*inch, 2*inch, 2*inch])
            contrib_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(contrib_table)
            elements.append(Spacer(1, 0.3*inch))
            
            # Allocation budgétaire
            elements.append(Paragraph("Allocation budgétaire optimisée", subtitle_style))
            
            if 'channel' in budget_df.columns:
                # Créer un tableau pour l'allocation budgétaire
                budget_data = [["Canal", "Budget (£)", "Budget (%)", "ROI"]]
                for _, row in budget_df.iterrows():
                    budget_data.append([
                        row['channel'],
                        f"{row['budget']:.2f}",
                        f"{row['budget_pct']:.2f}%",
                        f"{row['roi']:.2f}"
                    ])
                
                budget_table = Table(budget_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                budget_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(budget_table)
            else:
                elements.append(Paragraph("Données d'allocation budgétaire non disponibles.", normal_style))
            
            elements.append(Spacer(1, 0.3*inch))
                        
            # Recommandations
            elements.append(Paragraph("Recommandations", subtitle_style))
            elements.append(Spacer(1, 0.2*inch))
            
            best_roi_channel = "N/A"
            if 'channel' in budget_df.columns and 'roi' in budget_df.columns and not budget_df.empty:
                best_roi_channel = budget_df.loc[budget_df['roi'].idxmax()]['channel']
            
            # Définir un style de paragraphe personnalisé pour le texte de recommandation
            reco_style = ParagraphStyle(
                'RecoStyle',
                parent=normal_style,
                leftIndent=20,
                rightIndent=20,
                spaceBefore=10,
                spaceAfter=10,
                leading=14  # Espace entre les lignes
            )
            
            # Titre des recommandations
            elements.append(Paragraph("Basé sur notre analyse, voici nos recommandations:", normal_style))
            elements.append(Spacer(1, 0.1*inch))
            
            # Ajouter chaque recommandation comme un paragraphe séparé
            elements.append(Paragraph(f"1. Le canal '{best_roi_channel}' présente le meilleur ROI et devrait être privilégié.", reco_style))
            elements.append(Paragraph("2. L'allocation budgétaire optimisée présentée dans ce rapport permettrait d'améliorer le retour sur investissement global.", reco_style))
            elements.append(Paragraph("3. Une révision trimestrielle de l'allocation est recommandée pour s'adapter aux évolutions du marché.", reco_style))
            elements.append(Paragraph("4. Des tests A/B devraient être conduits pour valider empiriquement l'efficacité des différents canaux.", reco_style))
            
            # Générer le PDF
            doc.build(elements)
            
            # Télécharger le PDF
            st.sidebar.success("Rapport généré avec succès!")
            st.sidebar.download_button(
                label="Télécharger le rapport PDF",
                data=buffer.getvalue(),
                file_name="rapport_mmm.pdf",
                mime="application/pdf"
            )
            
        except Exception as e:
            st.sidebar.error(f"Erreur lors de la génération du rapport: {e}")
    else:
        st.sidebar.warning("Données non disponibles pour générer un rapport.")