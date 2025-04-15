# Script simple pour pusher des modifications vers GitHub
from google.colab import drive
import os

def push_to_github(project_path, branch="master", commit_message="Update project files"):
    """
    Fonction simple pour pusher vers GitHub
    
    Paramètres:
    - project_path: chemin vers votre projet (ex: '/content/drive/MyDrive/mmm-ecommerce')
    - branch: nom de la branche (par défaut: 'master')
    - commit_message: message du commit (par défaut: 'Update project files')
    """
    # Monter Google Drive
    print("🔄 Montage de Google Drive...")
    drive.mount('/content/drive')
    
    # Aller au dossier du projet
    print(f"📁 Accès au projet : {project_path}")
    os.chdir(project_path)
    
    # Voir l'état actuel
    print("\n🔍 État actuel du dépôt Git :")
    os.system("git status")
    
    # Ajouter les fichiers modifiés
    print("\n➕ Ajout des fichiers modifiés...")
    os.system("git add .")
    
    # Créer un commit
    print(f"\n💾 Création d'un commit: {commit_message}")
    os.system(f'git commit -m "{commit_message}"')
    
    # Pusher vers GitHub
    print(f"\n🚀 Envoi vers GitHub, branche '{branch}'...")
    os.system(f"git push origin {branch}")
    
    print("\n✅ Opération terminée!")

# Usage direct du script
if __name__ == "__main__":
    project_path = input("📁 Entrez le chemin vers votre projet: ")
    branch = input("🌿 Entrez le nom de la branche (défaut: master): ") or "master"
    commit_message = input("📝 Entrez votre message de commit: ") or "Update project files"
    
    push_to_github(project_path, branch, commit_message)
