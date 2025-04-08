# Script pour gérer les opérations Git depuis Google Colab
# Sauvegardez ce fichier dans votre dossier de projet

from google.colab import drive
import os
import sys

def setup_git():
    """Monte Google Drive et prépare l'environnement Git"""
    drive.mount('/content/drive')
    return True

def change_directory(project_path):
    """Change le répertoire vers le chemin du projet"""
    if os.path.exists(project_path):
        os.chdir(project_path)
        print(f"✅ Changement vers le répertoire: {project_path}")
        return True
    else:
        print(f"❌ Le chemin {project_path} n'existe pas")
        return False

def configure_git(email, name):
    """Configure les paramètres Git"""
    os.system(f"git config --global user.email {email}")
    os.system(f"git config --global user.name {name}")
    print("✅ Configuration Git effectuée")

def git_status():
    """Affiche l'état du dépôt Git"""
    print("\n🔍 État actuel du dépôt Git :")
    os.system("git status")

def add_files(specific_file=None):
    """Ajoute des fichiers à l'index Git"""
    if specific_file:
        os.system(f"git add {specific_file}")
        print(f"✅ Le fichier {specific_file} a été ajouté")
    else:
        os.system("git add .")
        print("✅ Tous les fichiers ont été ajoutés")

def create_commit(message):
    """Crée un nouveau commit"""
    if message:
        os.system(f'git commit -m "{message}"')
        print("✅ Commit créé avec succès")
    else:
        print("❌ Le message de commit ne peut pas être vide")

def push_to_github(branch, username=None, token=None, repo=None):
    """Push les modifications vers GitHub"""
    if username and token and repo:
        os.system(f"git push https://{username}:{token}@github.com/{repo}.git {branch}")
    else:
        os.system(f"git push origin {branch}")
    print("\n✅ Push terminé!")

def interactive_mode():
    """Mode interactif pour les opérations Git"""
    setup_git()
    
    # Chemin du projet
    project_path = input("📁 Entrez le chemin vers votre projet (ex: /content/drive/MyDrive/mmm-ecommerce): ")
    if not change_directory(project_path):
        return
    
    git_status()
    
    # Configuration Git
    configure = input("\n⚙️ Voulez-vous configurer Git? (oui/non): ").lower()
    if configure == 'oui':
        email = input("📧 Entrez votre email GitHub: ")
        name = input("👤 Entrez votre nom d'utilisateur GitHub: ")
        configure_git(email, name)
    
    # Ajout des fichiers
    add_choice = input("\n➕ Voulez-vous ajouter tous les fichiers modifiés? (oui/non): ").lower()
    if add_choice == 'oui':
        add_files()
    else:
        specific_file = input("📄 Entrez le chemin du fichier spécifique à ajouter: ")
        add_files(specific_file)
    
    # Création du commit
    commit_choice = input("\n💾 Voulez-vous créer un nouveau commit? (oui/non): ").lower()
    if commit_choice == 'oui':
        commit_message = input("📝 Entrez votre message de commit: ")
        create_commit(commit_message)
    
    # Push vers GitHub
    push_choice = input("\n🚀 Voulez-vous push vers GitHub? (oui/non): ").lower()
    if push_choice == 'oui':
        branch = input("🌿 Entrez le nom de votre branche (généralement 'master' ou 'main'): ")
        auth_method = input("🔑 Utiliser un token d'authentification? (oui/non): ").lower()
        
        if auth_method == 'oui':
            username = input("👤 Entrez votre nom d'utilisateur GitHub: ")
            token = input("🔐 Entrez votre token d'accès personnel GitHub: ")
            repo = input("📦 Entrez le nom de votre dépôt (ex: samirelais/MMM_EC): ")
            push_to_github(branch, username, token, repo)
        else:
            push_to_github(branch)

def quick_push(project_path, commit_message, branch="master"):
    """Mode rapide pour push des modifications"""
    setup_git()
    if change_directory(project_path):
        add_files()
        create_commit(commit_message)
        push_to_github(branch)

# Si le script est exécuté directement
if __name__ == "__main__":
    mode = input("Mode: (1) Interactif (2) Quick Push: ")
    if mode == "1":
        interactive_mode()
    elif mode == "2":
        project_path = input("Chemin du projet: ")
        commit_message = input("Message de commit: ")
        branch = input("Branche (défaut: master): ") or "master"
        quick_push(project_path, commit_message, branch)
    else:
        print("Mode non reconnu. Fin du script.")
