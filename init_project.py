#!/usr/bin/env python3
"""
Initialization script for the RNA 3D Folding project.
This script:
1. Checks and installs required dependencies
2. Sets up Kaggle API configuration
3. Creates necessary directories
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.11 or higher."""
    if sys.version_info < (3, 11):
        print("Error: Python 3.11 or higher is required.")
        sys.exit(1)
    print(f"✓ Python version: {sys.version}")

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
        print("✓ Dependencies installed successfully.")
    except subprocess.CalledProcessError:
        print("Error: Failed to install dependencies.")
        sys.exit(1)

def setup_kaggle_api():
    """Set up Kaggle API configuration."""
    # Check for kaggle.json in the project root
    project_root_kaggle_json = Path("kaggle.json")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"

    # If kaggle.json exists in the project root, use it
    if project_root_kaggle_json.exists():
        print(f"Found kaggle.json in project root. Using this file for authentication.")

        # Create .kaggle directory if it doesn't exist
        kaggle_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Copy the file to the default location
            with open(project_root_kaggle_json, 'r') as f:
                kaggle_config = json.load(f)

            with open(kaggle_json, 'w') as f:
                json.dump(kaggle_config, f)

            # Set permissions
            os.chmod(kaggle_json, 0o600)
            print("✓ Kaggle API configuration set up successfully.")
            return
        except Exception as e:
            print(f"Error copying kaggle.json: {e}")
            sys.exit(1)

    # Check if kaggle.json already exists in the default location
    if kaggle_json.exists():
        print("✓ Kaggle API configuration already exists in ~/.kaggle/")
        return

    # If no kaggle.json found, prompt the user
    kaggle_dir.mkdir(exist_ok=True)

    print("\nKaggle API setup:")
    print("1. Go to https://www.kaggle.com/account")
    print("2. Scroll down to 'API' section and click 'Create New API Token'")
    print("3. This will download a kaggle.json file")
    print("4. Enter the path to the downloaded kaggle.json file (or place it in the project root and run this script again):")

    kaggle_json_path = input("> ")
    kaggle_json_path = Path(kaggle_json_path.strip())

    if not kaggle_json_path.exists():
        print(f"Error: File {kaggle_json_path} does not exist.")
        sys.exit(1)

    try:
        # Copy the file
        with open(kaggle_json_path, 'r') as f:
            kaggle_config = json.load(f)

        with open(kaggle_json, 'w') as f:
            json.dump(kaggle_config, f)

        # Set permissions
        os.chmod(kaggle_json, 0o600)
        print("✓ Kaggle API configuration set up successfully.")
    except Exception as e:
        print(f"Error setting up Kaggle API: {e}")
        sys.exit(1)

def create_package_files():
    """Create necessary package files."""
    # Create __init__.py files in all directories
    for root, dirs, files in os.walk("src"):
        for dir_name in dirs:
            init_file = os.path.join(root, dir_name, "__init__.py")
            Path(init_file).touch(exist_ok=True)

    # Create __init__.py in src/rna_folding
    Path("src/rna_folding/__init__.py").touch(exist_ok=True)

    print("✓ Package files created.")

def main():
    """Main function to initialize the project."""
    print("Initializing RNA 3D Folding project...\n")

    check_python_version()
    create_package_files()
    install_dependencies()
    setup_kaggle_api()

    print("\nProject initialization completed successfully!")
    print("\nNext steps:")
    print("1. Run tests: pytest tests/")
    print("2. Download data: python src/rna_folding/data/download_data.py")

if __name__ == "__main__":
    main()
