import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import json
from datetime import datetime

class MMMVisualization:
   """
   Classe pour créer des visualisations pour le Marketing Mix Modeling.
   """
   def __init__(self, config_path):
        """
        Initialise la classe de visualisation avec un chemin de configuration.
        
        Args:
            config_path: Chemin vers le fichier de configuration JSON
        """
        # Charger la configuration
        with open(config_path, 'r') as config_file:
            self.config = json.load(config_file)
        
        # Créer le répertoire reports s'il n'existe pas
        os.makedirs("/content/drive/MyDrive/mmm-ecommerce/reports", exist_ok=True)
        os.makedirs("/content/drive/MyDrive/mmm-ecommerce/reports/figures", exist_ok=True)
        
        # Configuration du style de matplotlib
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use('default')  # Utiliser le style par défaut de matplotlib
            sns.set_theme()  # Utiliser le thème par défaut de Seaborn
        except ImportError:
            print("Matplotlib ou Seaborn non installé. Utilisation des paramètres par défaut.")
        
        # Palette de couleurs personnalisée pour les canaux
        self.channel_colors = {
            'tv': '#1f77b4',
            'radio': '#ff7f0e',
            'print': '#2ca02c',
            'social_media': '#d62728',
            'search': '#9467bd',
            'email': '#8c564b',
            'display': '#e377c2',
            'baseline': '#7f7f7f'
        }
   def plot_channel_contributions(self, contributions_df):
       """
       Crée un graphique des contributions par canal marketing.
       
       Args:
           contributions_df: DataFrame contenant les résultats d'attribution
           
       Returns:
           Chemin vers l'image sauvegardée
       """
       # Calculer la contribution moyenne par canal
       channel_contribs = {}
       channel_contribs['baseline'] = contributions_df['baseline_contribution'].mean()
       
       for channel in self.config['marketing_channels']:
           contrib_col = f"{channel}_contribution"
           if contrib_col in contributions_df.columns:
               channel_contribs[channel] = contributions_df[contrib_col].mean()
       
       # Créer un DataFrame pour le graphique
       contrib_df = pd.DataFrame({
           'channel': list(channel_contribs.keys()),
           'contribution': list(channel_contribs.values())
       })
       
       # Trier par contribution
       contrib_df = contrib_df.sort_values('contribution', ascending=False)
       
       # Créer le graphique
       plt.figure(figsize=(10, 6))
       bars = plt.bar(contrib_df['channel'], contrib_df['contribution'], 
                color=[self.channel_colors.get(ch, '#333333') for ch in contrib_df['channel']])
       
       # Ajouter des étiquettes
       plt.title('Contribution moyenne par canal', fontsize=14)
       plt.xlabel('Canal', fontsize=12)
       plt.ylabel('Contribution aux ventes (£)', fontsize=12)
       plt.xticks(rotation=45)
       
       # Ajouter les valeurs sur les barres
       for bar in bars:
           height = bar.get_height()
           plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'£{height:.0f}',
                   ha='center', va='bottom', fontsize=10)
       
       plt.tight_layout()
       
       # Sauvegarder le graphique
       output_path = "/content/drive/MyDrive/mmm-ecommerce/reports/figures/channel_contributions.png"
       plt.savefig(output_path, dpi=300)
       plt.close()
       
       return output_path
   
   def plot_roi_by_channel(self, contributions_df):
       """
       Crée un graphique du ROI par canal marketing.
       
       Args:
           contributions_df: DataFrame contenant les résultats d'attribution et les dépenses
           
       Returns:
           Chemin vers l'image sauvegardée
       """
       # Calculer le ROI médian par canal
       roi_data = {}
       
       for channel in self.config['marketing_channels']:
           roi_col = f"{channel}_roi"
           if roi_col in contributions_df.columns:
               roi_data[channel] = contributions_df[roi_col].median()
       
       # Créer un DataFrame pour le graphique
       roi_df = pd.DataFrame({
           'channel': list(roi_data.keys()),
           'roi': list(roi_data.values())
       })
       
       # Trier par ROI
       roi_df = roi_df.sort_values('roi', ascending=False)
       
       # Créer le graphique
       plt.figure(figsize=(10, 6))
       bars = plt.bar(roi_df['channel'], roi_df['roi'], 
                color=[self.channel_colors.get(ch, '#333333') for ch in roi_df['channel']])
       
       # Ajouter des étiquettes
       plt.title('ROI médian par canal', fontsize=14)
       plt.xlabel('Canal', fontsize=12)
       plt.ylabel('ROI (£ générés par £ dépensée)', fontsize=12)
       plt.xticks(rotation=45)
       
       # Ajouter les valeurs sur les barres
       for bar in bars:
           height = bar.get_height()
           plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)
       
       plt.tight_layout()
       
       # Sauvegarder le graphique
       output_path = "/content/drive/MyDrive/mmm-ecommerce/reports/figures/roi_by_channel.png"
       plt.savefig(output_path, dpi=300)
       plt.close()
       
       return output_path
   
   def plot_budget_allocation(self, budget_df):
       """
       Crée un graphique de l'allocation budgétaire optimisée.
       
       Args:
           budget_df: DataFrame contenant l'allocation budgétaire
           
       Returns:
           Chemin vers l'image sauvegardée
       """
       # Trier par budget
       sorted_df = budget_df.sort_values('budget', ascending=False)
       
       # Créer le graphique
       fig, ax1 = plt.subplots(figsize=(12, 7))
       
       # Graphique en barres pour le budget
       bars = ax1.bar(sorted_df['channel'], sorted_df['budget'], 
               color=[self.channel_colors.get(ch, '#333333') for ch in sorted_df['channel']])
       
       # Ajouter les étiquettes pour le budget
       ax1.set_xlabel('Canal', fontsize=12)
       ax1.set_ylabel('Budget alloué (£)', fontsize=12)
       ax1.tick_params(axis='x', rotation=45)
       
       # Ajouter les valeurs du budget sur les barres
       for bar in bars:
           height = bar.get_height()
           ax1.text(bar.get_x() + bar.get_width()/2., height + 500,
                  f'£{height:.0f}',
                  ha='center', va='bottom', fontsize=10)
       
       # Créer un deuxième axe Y pour le ROI
       ax2 = ax1.twinx()
       
       # Tracer le ROI comme une ligne
       ax2.plot(sorted_df['channel'], sorted_df['roi'], 'ro-', linewidth=2, markersize=8)
       ax2.set_ylabel('ROI', fontsize=12, color='r')
       ax2.tick_params(axis='y', labelcolor='r')
       
       # Ajouter un titre
       plt.title('Allocation budgétaire optimisée et ROI par canal', fontsize=14)
       
       # Ajouter une légende
       lines, labels = ax1.get_legend_handles_labels()
       lines2, labels2 = ax2.get_legend_handles_labels()
       ax2.legend(lines + lines2, ['Budget', 'ROI'], loc='upper right')
       
       plt.tight_layout()
       
       # Sauvegarder le graphique
       output_path = "/content/drive/MyDrive/mmm-ecommerce/reports/figures/budget_allocation.png"
       plt.savefig(output_path, dpi=300)
       plt.close()
       
       return output_path
   
   def plot_actual_vs_predicted(self, full_df, model):
       """
       Crée un graphique des ventes réelles vs prédites dans le temps.
       
       Args:
           full_df: DataFrame Spark complet avec les données
           model: Modèle MMM entraîné
           
       Returns:
           Chemin vers l'image sauvegardée
       """
       # Convertir en pandas  
       pdf = full_df.toPandas()
       
       # Préparer les données pour la prédiction
       X = pdf.drop(['date', 'revenue'], axis=1)
       
       # Faire les prédictions
       y_pred = model.predict(X)
       
       # Créer un DataFrame avec les dates, ventes réelles et prédites
       result_df = pd.DataFrame({
           'date': pd.to_datetime(pdf['date']),
           'actual': pdf['revenue'],
           'predicted': y_pred
       })
       
       # Trier par date
       result_df = result_df.sort_values('date')
       
       # Créer le graphique
       plt.figure(figsize=(14, 7))
       plt.plot(result_df['date'], result_df['actual'], 'b-', label='Ventes réelles')
       plt.plot(result_df['date'], result_df['predicted'], 'r-', label='Ventes prédites')
       
       # Ajouter la légende et les étiquettes
       plt.legend(fontsize=12)
       plt.title('Ventes réelles vs prédites', fontsize=14)
       plt.xlabel('Date', fontsize=12)
       plt.ylabel('Ventes (£)', fontsize=12)
       plt.grid(True)
       plt.xticks(rotation=45)
       
       # Sauvegarder le graphique
       output_path = "/content/drive/MyDrive/mmm-ecommerce/reports/figures/actual_vs_predicted.png"
       plt.savefig(output_path, dpi=300, bbox_inches='tight')
       plt.close()
       
       return output_path
   
   def plot_contributions_over_time(self, contributions_df):
       """
       Crée un graphique des contributions par canal dans le temps.
       
       Args:
           contributions_df: DataFrame contenant les résultats d'attribution
           
       Returns:
           Chemin vers l'image sauvegardée
       """
       # Préparer les données
       plot_df = contributions_df.copy()
       plot_df['date'] = pd.to_datetime(plot_df['date'])
       
       # Extraire les colonnes de contribution
       contrib_columns = [f"{channel}_contribution" for channel in self.config['marketing_channels']
                         if f"{channel}_contribution" in plot_df.columns]
       contrib_columns.append('baseline_contribution')
       
       # Créer un DataFrame au format long pour faciliter le traçage
       melted_df = pd.melt(plot_df, id_vars=['date'], value_vars=contrib_columns,
                          var_name='channel', value_name='contribution')
       
       # Nettoyer les noms de canaux (enlever le suffixe "_contribution")
       melted_df['channel'] = melted_df['channel'].str.replace('_contribution', '')
       
       # Créer le graphique
       plt.figure(figsize=(14, 8))
       
       # Utiliser une palette de couleurs personnalisée
       colors = [self.channel_colors.get(ch.replace('_contribution', ''), '#333333') 
                for ch in contrib_columns]
       
       # Tracer les séries temporelles empilées
       ax = sns.lineplot(x='date', y='contribution', hue='channel', data=melted_df,
                       palette=colors, linewidth=2)
       
       # Ajouter les étiquettes
       plt.title('Contributions des canaux marketing dans le temps', fontsize=14)
       plt.xlabel('Date', fontsize=12)
       plt.ylabel('Contribution aux ventes (£)', fontsize=12)
       plt.xticks(rotation=45)
       
       # Améliorer la légende
       plt.legend(title='Canal', fontsize=10, title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')
       
       plt.tight_layout()
       
       # Sauvegarder le graphique
       output_path = "/content/drive/MyDrive/mmm-ecommerce/reports/figures/contributions_over_time.png"
       plt.savefig(output_path, dpi=300, bbox_inches='tight')
       plt.close()
       
       return output_path
   
   def generate_report(self, metrics, contributions_df, feature_importances, budget_allocation):
       """
       Génère un rapport complet en markdown avec les résultats de l'analyse MMM.
       
       Args:
           metrics: Dictionnaire des métriques d'évaluation
           contributions_df: DataFrame des contributions par canal
           feature_importances: DataFrame des importances des caractéristiques
           budget_allocation: DataFrame de l'allocation budgétaire optimisée
           
       Returns:
           Chemin vers le rapport généré
       """
       # Créer le contenu du rapport
       report = []
       
       # En-tête
       report.append("# Rapport d'analyse Marketing Mix Modeling (MMM)")
       report.append(f"*Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}*\n")
       
       # Résumé exécutif
       report.append("## Résumé exécutif")
       report.append("Cette analyse utilise le Marketing Mix Modeling pour évaluer l'efficacité des différents canaux marketing et optimiser l'allocation budgétaire.")
       
       # Métriques du modèle
       report.append("\n## Performance du modèle")
       report.append(f"- **R² (coefficient de détermination)**: {metrics['r2']:.4f}")
       report.append(f"- **RMSE (erreur quadratique moyenne)**: {metrics['rmse']:.2f}")
       report.append(f"- **MAE (erreur absolue moyenne)**: {metrics['mae']:.2f}")
       report.append(f"- **MAPE (erreur en pourcentage absolue moyenne)**: {metrics['mape']:.2f}%\n")
       
       # Ajouter le graphique de prédiction
       report.append("### Ventes réelles vs prédites")
       report.append("![Ventes réelles vs prédites](figures/actual_vs_predicted.png)\n")
       
       # Contributions des canaux
       report.append("## Contributions des canaux marketing")
       
       # Calculer la contribution moyenne par canal
       channel_contribs = {}
       channel_contribs['baseline'] = contributions_df['baseline_contribution'].mean()
       
       for channel in self.config['marketing_channels']:
           contrib_col = f"{channel}_contribution"
           if contrib_col in contributions_df.columns:
               channel_contribs[channel] = contributions_df[contrib_col].mean()
       
       # Créer un DataFrame pour afficher les contributions
       contrib_df = pd.DataFrame({
           'canal': list(channel_contribs.keys()),
           'contribution_moyenne': list(channel_contribs.values()),
           'contribution_pourcentage': [v / contributions_df['predicted_revenue'].mean() * 100 for v in channel_contribs.values()]
       }).sort_values('contribution_moyenne', ascending=False)
       
       # Ajouter un tableau des contributions
       report.append("### Contributions moyennes par canal")
       report.append(contrib_df.to_markdown(index=False, floatfmt=".2f"))
       report.append("")
       
       # Ajouter le graphique des contributions
       report.append("### Visualisation des contributions")
       report.append("![Contributions par canal](figures/channel_contributions.png)\n")
       
       # ROI par canal
       report.append("## Retour sur investissement (ROI) par canal")
       
       # Calculer le ROI médian par canal
       roi_data = {}
       for channel in self.config['marketing_channels']:
           roi_col = f"{channel}_roi"
           if roi_col in contributions_df.columns:
               roi_data[channel] = contributions_df[roi_col].median()
       
       # Créer un DataFrame pour afficher le ROI
       roi_df = pd.DataFrame({
           'canal': list(roi_data.keys()),
           'roi_median': list(roi_data.values())
       }).sort_values('roi_median', ascending=False)
       
       # Ajouter un tableau du ROI
       report.append("### ROI médian par canal")
       report.append(roi_df.to_markdown(index=False, floatfmt=".2f"))
       report.append("")
       
       # Ajouter le graphique du ROI
       report.append("### Visualisation du ROI")
       report.append("![ROI par canal](figures/roi_by_channel.png)\n")
       
       # Allocation budgétaire optimisée
       report.append("## Allocation budgétaire optimisée")
       
       # Formatter le DataFrame d'allocation
       budget_df = budget_allocation.copy()
       budget_df.columns = ['Canal', 'Budget (£)', 'Budget (%)', 'ROI']
       
       # Ajouter un tableau d'allocation
       report.append("### Répartition recommandée du budget")
       report.append(budget_df.to_markdown(index=False, floatfmt=".2f"))
       report.append("")
       
       # Ajouter le graphique d'allocation
       report.append("### Visualisation de l'allocation budgétaire")
       report.append("![Allocation budgétaire](figures/budget_allocation.png)\n")
       
       # Importance des caractéristiques
       report.append("## Importance des caractéristiques")
       
       # Formatting for feature importance table
       top_features = feature_importances.head(15).copy()
       top_features.columns = ['Caractéristique', 'Importance']
       
       # Add feature importance table
       report.append("### Top 15 des caractéristiques les plus importantes")
       report.append(top_features.to_markdown(index=False, floatfmt=".2f"))
       report.append("")
       
       # Conclusions et recommandations
       report.append("## Conclusions et recommandations")
       
       # Trouver les canaux les plus performants
       top_roi_channels = roi_df.head(3)['canal'].tolist()
       bottom_roi_channels = roi_df.tail(3)['canal'].tolist()
       
       report.append("### Principaux enseignements")
       report.append(f"- Les canaux avec le meilleur ROI sont: **{', '.join(top_roi_channels)}**.")
       report.append(f"- Les canaux avec le ROI le plus faible sont: **{', '.join(bottom_roi_channels)}**.")
       report.append("- La contribution de base (non attribuée aux canaux marketing) représente "
                    f"**{contrib_df[contrib_df['canal'] == 'baseline']['contribution_pourcentage'].values[0]:.1f}%** des ventes.")
       
       report.append("\n### Recommandations")
       report.append("1. **Réallocation budgétaire**: Ajuster le budget marketing selon les recommandations d'allocation optimisée.")
       report.append(f"2. **Augmenter les investissements**: Envisager d'augmenter les investissements dans les canaux à haut ROI comme **{', '.join(top_roi_channels[:2])}**.")
       report.append(f"3. **Optimiser ou réduire**: Revoir la stratégie pour les canaux à faible ROI comme **{', '.join(bottom_roi_channels[:2])}**.")
       report.append("4. **Tests supplémentaires**: Réaliser des tests A/B pour valider l'efficacité des canaux recommandés.")
       report.append("5. **Analyse saisonnière**: Adapter la répartition du budget en fonction des variations saisonnières observées.")
       
       # Méthode et limitations
       report.append("\n## Méthodologie et limitations")
       report.append("### Méthodologie")
       report.append("Cette analyse utilise un modèle LightGBM avec des transformations d'adstock et de saturation pour modéliser les effets marketing. "
                    "Les données sont divisées en ensembles d'entraînement et de test chronologiques.")
       
       report.append("\n### Limitations")
       report.append("- Les dépenses marketing sont simulées sur la base des patterns de vente observés.")
       report.append("- Le modèle ne prend pas en compte toutes les interactions possibles entre les canaux.")
       report.append("- D'autres facteurs externes non mesurés peuvent influencer les ventes.")
       
       # Joindre le rapport en un seul texte
       report_text = "\n".join(report)
       
       # Sauvegarder le rapport au format markdown
       output_path = "/content/drive/MyDrive/mmm-ecommerce/reports/mmm_report.md"
       with open(output_path, 'w') as f:
           f.write(report_text)
       
       # Si la génération de rapports HTML est activée
       if self.config['reporting'].get('generate_html_report', True):
           try:
               import markdown
               html = markdown.markdown(report_text, extensions=['tables'])
               
               # Ajouter du style CSS
               html_template = f"""
               <!DOCTYPE html>
               <html>
               <head>
                   <title>Rapport MMM</title>
                   <style>
                       body {{ font-family: Arial, sans-serif; line-height: 1.6; max-width: 1200px; margin: 0 auto; padding: 20px; }}
                       h1, h2, h3 {{ color: #2c3e50; }}
                       table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                       th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                       th {{ background-color: #f2f2f2; }}
                       img {{ max-width: 100%; height: auto; }}
                       .container {{ background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                   </style>
               </head>
               <body> 
                   <div class="container">
                   {html}
                   </div>
               </body>
               </html>
               """
               
               # Sauvegarder le rapport au format HTML
               html_path = "/content/drive/MyDrive/mmm-ecommerce/reports/mmm_report.html"
               with open(html_path, 'w') as f:
                   f.write(html_template)
               
               print(f"Rapport HTML généré: {html_path}")
               
           except ImportError:
               print("Le module 'markdown' n'est pas installé. Rapport HTML non généré.")
       
       return output_path
