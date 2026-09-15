import os
import shutil
from git import Repo


def clone_repository(repo_url, destination="data/repository"):
    """
    Clone a GitHub repository into the destination folder.
    """

    # If the folder already exists, remove it
    if os.path.exists(destination):
        shutil.rmtree(destination)

    print("📥 Cloning repository...")
    print(f"🔗 URL: {repo_url}")

    try:
        Repo.clone_from(repo_url, destination)

        print("\n✅ Repository cloned successfully!")
        print(f"📁 Location: {destination}")

        return destination

    except Exception as e:
        print("\n Failed to clone repository.")
        print(f"Error: {e}")

        return None


if __name__ == "__main__":

    repo_url = input("Enter GitHub repository URL: ").strip()

    if not repo_url:
        print(" Please enter a repository URL.")
    else:
        clone_repository(repo_url)