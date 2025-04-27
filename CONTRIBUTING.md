# Contributing to RNA 3D Structure Prediction

Thank you for your interest in contributing to the RNA 3D Structure Prediction project! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## How to Contribute

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes
4. Run tests to ensure your changes don't break existing functionality
5. Submit a pull request

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stanford-rna-3d-solving.git
   cd stanford-rna-3d-solving
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -e .
   ```

4. Run tests:
   ```bash
   ./run_rna_pipeline.sh test
   ```

## Project Structure

Please follow the existing project structure when adding new files or modules:

- Place model implementations in `src/rna_folding/models/`
- Place data handling code in `src/rna_folding/data/`
- Place evaluation code in `src/rna_folding/evaluation/`
- Place visualization code in `src/rna_folding/visualization/`

## Coding Style

- Follow PEP 8 guidelines for Python code
- Use docstrings to document functions and classes
- Write unit tests for new functionality
- Keep functions small and focused on a single task
- Use meaningful variable and function names

## Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the documentation if necessary
3. The PR should work on all supported platforms
4. The PR will be merged once it has been reviewed and approved

## Questions?

If you have any questions about contributing, please open an issue in the repository.
