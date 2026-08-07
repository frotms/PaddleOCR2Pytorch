"""
Shared pytest fixtures and configuration for all tests.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
import torch
import numpy as np
from PIL import Image


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a sample image for testing."""
    img_path = temp_dir / "test_image.jpg"
    img = Image.new('RGB', (640, 480), color='white')
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_numpy_image() -> np.ndarray:
    """Create a sample numpy array image."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_tensor_image() -> torch.Tensor:
    """Create a sample torch tensor image."""
    return torch.rand(3, 480, 640)


@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Create a mock configuration dictionary."""
    return {
        "Architecture": {
            "model_type": "det",
            "algorithm": "DB",
            "Transform": None,
            "Backbone": {
                "name": "MobileNetV3",
                "scale": 0.5,
                "model_name": "large",
            },
            "Neck": {
                "name": "DBFPN",
                "out_channels": 256,
            },
            "Head": {
                "name": "DBHead",
                "k": 50,
            },
        },
        "Global": {
            "use_gpu": False,
            "epoch_num": 500,
            "log_smooth_window": 20,
            "print_batch_step": 10,
            "save_model_dir": "./output/db_mv3/",
            "save_epoch_step": 10,
            "eval_batch_step": [0, 100],
            "pretrained_model": None,
            "checkpoints": None,
            "save_inference_dir": None,
        },
    }


@pytest.fixture
def mock_yaml_config(temp_dir: Path, mock_config: Dict[str, Any]) -> Path:
    """Create a mock YAML configuration file."""
    import yaml
    
    config_path = temp_dir / "config.yml"
    with open(config_path, 'w') as f:
        yaml.dump(mock_config, f)
    return config_path


@pytest.fixture
def mock_model_path(temp_dir: Path) -> Path:
    """Create a mock model file path."""
    model_path = temp_dir / "model.pth"
    torch.save({"state_dict": {}, "config": {}}, model_path)
    return model_path


@pytest.fixture
def sample_text_file(temp_dir: Path) -> Path:
    """Create a sample text file."""
    text_path = temp_dir / "sample.txt"
    text_path.write_text("Sample text for testing\nLine 2\nLine 3")
    return text_path


@pytest.fixture
def sample_dict_file(temp_dir: Path) -> Path:
    """Create a sample dictionary file for character recognition."""
    dict_path = temp_dir / "dict.txt"
    chars = ['a', 'b', 'c', 'd', 'e', '0', '1', '2', '3', '4', ' ']
    dict_path.write_text('\n'.join(chars))
    return dict_path


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)


@pytest.fixture
def gpu_available() -> bool:
    """Check if GPU is available for testing."""
    return torch.cuda.is_available()


@pytest.fixture
def mock_db_config() -> Dict[str, Any]:
    """Mock configuration for DB text detection."""
    return {
        "thresh": 0.3,
        "box_thresh": 0.6,
        "max_candidates": 1000,
        "unclip_ratio": 1.5,
        "use_dilation": False,
        "score_mode": "fast",
    }


@pytest.fixture
def mock_rec_config() -> Dict[str, Any]:
    """Mock configuration for text recognition."""
    return {
        "character_dict_path": "./ppocr/utils/ppocr_keys_v1.txt",
        "use_space_char": True,
        "max_text_length": 25,
        "limited_max_width": 1280,
        "limited_min_width": 16,
    }


@pytest.fixture
def cleanup_cuda():
    """Cleanup CUDA cache after tests."""
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )