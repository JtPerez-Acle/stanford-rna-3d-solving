"""
Module for downloading competition data from Kaggle.
"""

import os
import subprocess
import json
from pathlib import Path
import zipfile
import shutil
from tqdm import tqdm

def check_kaggle_api():
    """
    Check if Kaggle API is properly configured.

    Returns:
        bool: True if Kaggle API is configured, False otherwise.
    """
    # First, check if kaggle.json exists in the project root
    project_root_kaggle_json = Path("kaggle.json")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    # If kaggle.json exists in the project root, copy it to the default location
    if project_root_kaggle_json.exists():
        print(f"Found kaggle.json in project root. Using this file for authentication.")

        # Create .kaggle directory if it doesn't exist
        kaggle_dir.mkdir(parents=True, exist_ok=True)

        # Copy the file to the default location
        try:
            with open(project_root_kaggle_json, 'r') as f:
                kaggle_config = json.load(f)

            with open(kaggle_json, 'w') as f:
                json.dump(kaggle_config, f)

            # Set permissions
            os.chmod(kaggle_json, 0o600)
        except Exception as e:
            print(f"Error copying kaggle.json: {e}")
            return False
    elif not kaggle_json.exists():
        print("Kaggle API configuration not found.")
        print("Please place your kaggle.json file in the project root or in ~/.kaggle/")
        return False

    try:
        # Test Kaggle API by listing competitions
        result = subprocess.run(
            ["kaggle", "competitions", "list", "-s", "stanford"],
            capture_output=True,
            text=True,
            check=True
        )
        return "stanford-rna-3d-folding" in result.stdout
    except subprocess.CalledProcessError:
        print("Failed to connect to Kaggle API.")
        return False
    except FileNotFoundError:
        print("Kaggle command not found. Please install the Kaggle package.")
        return False

def get_competition_files(competition_name):
    """
    Get list of files available in the competition.

    Args:
        competition_name (str): Name of the Kaggle competition.

    Returns:
        list: List of file names in the competition.
    """
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "files", "-v", competition_name],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse the output to get file names
        lines = result.stdout.strip().split('\n')
        files = [line.split()[-1] for line in lines if line and not line.startswith('ref')]
        return files
    except subprocess.CalledProcessError as e:
        print(f"Failed to get competition files: {e}")
        return []

def download_competition_data(competition_name, data_dir, files=None):
    """
    Download competition data from Kaggle.

    Args:
        competition_name (str): Name of the Kaggle competition.
        data_dir (Path): Directory to save the data.
        files (list, optional): List of specific files to download. If None, download all files.

    Returns:
        bool: True if download was successful, False otherwise.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create a temporary directory for downloads
    temp_dir = data_dir / "temp"
    temp_dir.mkdir(exist_ok=True)

    try:
        # Download competition data
        print(f"Downloading data for competition: {competition_name}")
        cmd = ["kaggle", "competitions", "download", "-c", competition_name, "-p", str(temp_dir)]

        if files:
            for file in files:
                file_cmd = cmd + ["-f", file]
                subprocess.run(file_cmd, check=True)
        else:
            subprocess.run(cmd, check=True)

        # Extract the downloaded files
        extract_competition_files(temp_dir, data_dir)

        # Clean up temporary directory
        shutil.rmtree(temp_dir)

        print(f"Data downloaded successfully to {data_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to download competition data: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def extract_competition_files(source_dir, target_dir):
    """
    Extract competition files from zip archives.

    Args:
        source_dir (Path): Directory containing downloaded zip files.
        target_dir (Path): Directory to extract files to.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    # Find all zip files
    zip_files = list(source_dir.glob("*.zip"))

    for zip_file in tqdm(zip_files, desc="Extracting files"):
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

    # Also copy any non-zip files
    for file in source_dir.glob("*"):
        if file.is_file() and not file.name.endswith(".zip"):
            shutil.copy(file, target_dir / file.name)

def main():
    """Main function to download competition data."""
    if not check_kaggle_api():
        return

    competition_name = "stanford-rna-3d-folding"
    data_dir = Path("data/raw")

    download_competition_data(competition_name, data_dir)

if __name__ == "__main__":
    main()
