# Rapport d'analyse Marketing Mix Modeling (MMM)
*Généré le 13/04/2025 à 21:09*

## Résumé exécutif
Cette analyse utilise le Marketing Mix Modeling pour évaluer l'efficacité des différents canaux marketing et optimiser l'allocation budgétaire.

## Performance du modèle
- **R² (coefficient de détermination)**: -0.0906
- **RMSE (erreur quadratique moyenne)**: 28208.83
- **MAE (erreur absolue moyenne)**: 16805.72
- **MAPE (erreur en pourcentage absolue moyenne)**: 33.63%

### Ventes réelles vs prédites
![Ventes réelles vs prédites](figures/actual_vs_predicted.png)

## Contributions des canaux marketing
### Contributions moyennes par canal
| canal        |   contribution_moyenne |   contribution_pourcentage |
|:-------------|-----------------------:|---------------------------:|
| baseline     |               41074.92 |                     119.32 |
| tv           |                4082.64 |                      11.86 |
| display      |                -116.01 |                      -0.34 |
| print        |                -268.12 |                      -0.78 |
| email        |                -512.88 |                      -1.49 |
| search       |               -1023.01 |                      -2.97 |
| social_media |               -2146.06 |                      -6.23 |
| radio        |               -6667.54 |                     -19.37 |

### Visualisation des contributions
![Contributions par canal](figures/channel_contributions.png)

## Retour sur investissement (ROI) par canal
### ROI médian par canal
| canal        |   roi_median |
|:-------------|-------------:|
| tv           |         3.61 |
| display      |        -0.01 |
| email        |        -0.01 |
| print        |        -0.05 |
| search       |        -0.11 |
| social_media |        -0.63 |
| radio        |        -2.98 |

### Visualisation du ROI
![ROI par canal](figures/roi_by_channel.png)

## Allocation budgétaire optimisée
### Répartition recommandée du budget
| Canal        |   Budget (£) |   Budget (%) |   ROI |
|:-------------|-------------:|-------------:|------:|
| tv           |     40000.00 |        40.00 |  3.61 |
| display      |      9285.71 |         9.29 | -0.01 |
| email        |      9285.71 |         9.29 | -0.01 |
| print        |      9285.71 |         9.29 | -0.05 |
| search       |      9285.71 |         9.29 | -0.11 |
| social_media |      9285.71 |         9.29 | -0.63 |
| radio        |      9285.71 |         9.29 | -2.98 |

### Visualisation de l'allocation budgétaire
![Allocation budgétaire](figures/budget_allocation.png)

## Importance des caractéristiques
### Top 15 des caractéristiques les plus importantes
| Caractéristique      |      Importance |
|:---------------------|----------------:|
| transactions         | 207309111852.00 |
| unique_customers     |  53127668428.00 |
| day_of_week          |  40870127848.00 |
| consumer_confidence  |  32636134072.00 |
| radio_adstock        |  32053177818.00 |
| radio                |  27002654004.00 |
| gdp_growth           |  18671990156.00 |
| social_media_adstock |  16407048356.00 |
| display_adstock      |  16163602440.00 |
| print_adstock        |  15709270184.00 |
| tv                   |  14511311664.00 |
| tv_adstock           |  14114973578.00 |
| day_of_week_cos      |  13541231644.00 |
| print                |  12386248004.00 |
| month                |  12193807374.00 |

## Conclusions et recommandations
### Principaux enseignements
- Les canaux avec le meilleur ROI sont: **tv, display, email**.
- Les canaux avec le ROI le plus faible sont: **search, social_media, radio**.
- La contribution de base (non attribuée aux canaux marketing) représente **119.3%** des ventes.

### Recommandations
1. **Réallocation budgétaire**: Ajuster le budget marketing selon les recommandations d'allocation optimisée.
2. **Augmenter les investissements**: Envisager d'augmenter les investissements dans les canaux à haut ROI comme **tv, display**.
3. **Optimiser ou réduire**: Revoir la stratégie pour les canaux à faible ROI comme **search, social_media**.
4. **Tests supplémentaires**: Réaliser des tests A/B pour valider l'efficacité des canaux recommandés.
5. **Analyse saisonnière**: Adapter la répartition du budget en fonction des variations saisonnières observées.

## Méthodologie et limitations
### Méthodologie
Cette analyse utilise un modèle LightGBM avec des transformations d'adstock et de saturation pour modéliser les effets marketing. Les données sont divisées en ensembles d'entraînement et de test chronologiques.

### Limitations
- Les dépenses marketing sont simulées sur la base des patterns de vente observés.
- Le modèle ne prend pas en compte toutes les interactions possibles entre les canaux.
- D'autres facteurs externes non mesurés peuvent influencer les ventes.