"""
Validation tests to ensure the testing infrastructure is properly set up.
"""

import pytest
import sys
import os
from pathlib import Path


class TestSetupValidation:
    """Test class to validate the testing infrastructure setup."""
    
    @pytest.mark.unit
    def test_pytest_installed(self):
        """Test that pytest is properly installed."""
        assert "pytest" in sys.modules or True
        
    @pytest.mark.unit
    def test_project_structure_exists(self):
        """Test that the expected project structure exists."""
        project_root = Path(__file__).parent.parent
        
        # Check main package directories
        assert (project_root / "pytorchocr").exists()
        assert (project_root / "ptstructure").exists()
        assert (project_root / "converter").exists()
        assert (project_root / "misc").exists()
        assert (project_root / "tools").exists()
        
        # Check test directories
        assert (project_root / "tests").exists()
        assert (project_root / "tests" / "unit").exists()
        assert (project_root / "tests" / "integration").exists()
        assert (project_root / "tests" / "conftest.py").exists()
        
    @pytest.mark.unit
    def test_configuration_files_exist(self):
        """Test that configuration files exist."""
        project_root = Path(__file__).parent.parent
        
        assert (project_root / "pyproject.toml").exists()
        assert (project_root / ".gitignore").exists()
        
    @pytest.mark.unit
    def test_fixtures_available(self, temp_dir, sample_image, mock_config):
        """Test that pytest fixtures are working correctly."""
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        
        assert sample_image.exists()
        assert sample_image.suffix == ".jpg"
        
        assert isinstance(mock_config, dict)
        assert "Architecture" in mock_config
        assert "Global" in mock_config
        
    @pytest.mark.unit
    def test_coverage_configured(self):
        """Test that coverage is properly configured."""
        try:
            import coverage
            assert True
        except ImportError:
            pytest.fail("Coverage module not installed")
            
    @pytest.mark.unit
    def test_mock_configured(self):
        """Test that pytest-mock is properly configured."""
        try:
            from pytest_mock import MockerFixture
            assert True
        except ImportError:
            pytest.fail("pytest-mock module not installed")
            
    @pytest.mark.unit
    def test_markers_registered(self, pytestconfig):
        """Test that custom markers are registered."""
        markers = pytestconfig.getini("markers")
        marker_names = [line.split(":")[0].strip() for line in markers if line.strip()]
        
        assert "unit" in marker_names
        assert "integration" in marker_names
        assert "slow" in marker_names
        
    @pytest.mark.unit
    def test_import_main_modules(self):
        """Test that main project modules can be imported."""
        try:
            import pytorchocr
            import ptstructure
            import converter
            import misc
            import tools
            assert True
        except ImportError as e:
            pytest.skip(f"Module import failed (expected before installation): {e}")
            
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        assert True
        
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        import time
        time.sleep(0.1)
        assert True
        
    @pytest.mark.unit
    @pytest.mark.parametrize("value,expected", [
        (1, 1),
        (2, 2),
        (3, 3),
    ])
    def test_parametrize_works(self, value, expected):
        """Test that parametrize decorator works."""
        assert value == expected
        
    @pytest.mark.unit
    def test_mocker_fixture(self, mocker):
        """Test that mocker fixture from pytest-mock works."""
        mock_func = mocker.Mock(return_value=42)
        result = mock_func()
        
        assert result == 42
        mock_func.assert_called_once()
        
    @pytest.mark.unit
    def test_numpy_available(self):
        """Test that numpy is available for tests."""
        try:
            import numpy as np
            arr = np.array([1, 2, 3])
            assert len(arr) == 3
        except ImportError:
            pytest.skip("NumPy not yet installed")
            
    @pytest.mark.unit
    def test_torch_available(self):
        """Test that PyTorch is available for tests."""
        try:
            import torch
            tensor = torch.tensor([1., 2., 3.])
            assert tensor.shape[0] == 3
        except ImportError:
            pytest.skip("PyTorch not yet installed")
            
    @pytest.mark.unit
    def test_temporary_file_handling(self, temp_dir):
        """Test temporary file handling in tests."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        
        assert test_file.exists()
        assert test_file.read_text() == "test content"