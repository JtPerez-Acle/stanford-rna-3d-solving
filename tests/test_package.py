"""
Basic tests to ensure the package structure is working correctly.
"""

import pytest
import importlib

def test_package_imports():
    """Test that all package modules can be imported."""
    modules = [
        'rna_folding',
        'rna_folding.data',
        'rna_folding.data.download',
        'rna_folding.data.analysis',
        'rna_folding.visualization',
        'rna_folding.visualization.visualize',
        'rna_folding.models',
        'rna_folding.models.base',
        'rna_folding.models.data',
        'rna_folding.models.metrics',
        'rna_folding.models.multi_scale',
        'rna_folding.models.physics',
        'rna_folding.models.train',
        'rna_folding.models.predict',
    ]

    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_name}: {e}")
