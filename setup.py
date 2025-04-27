from setuptools import setup, find_packages

setup(
    name="rna_folding",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "biopython",
        "pytest",
        "kaggle",
        "py3Dmol",  # For 3D visualization of RNA structures
        "nglview",  # Alternative for 3D visualization
        "ipywidgets",  # Required for interactive visualizations in notebooks
        "tqdm",  # For progress bars
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pytest-cov",
            "jupyter",
        ],
    },
)
