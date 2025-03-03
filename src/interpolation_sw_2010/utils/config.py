"""
Configuration management for the Stock-Watson interpolation procedure.
"""

import yaml
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)

class Config:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            config_path = Path(__file__).parent / "config.yaml"
            if not config_path.exists():
                self._create_default_config(config_path)
            
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self._config = self._get_default_config()
    
    def _create_default_config(self, config_path: Path):
        """Create default configuration file if it doesn't exist."""
        config = self._get_default_config()
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "data": {
                "default_voa": [1, 4, 5, 0, 2, 4, 4, 7, 1],
                "input_file": "data/raw_data.xlsx",
                "output_file": "data/transformed_data.xlsx"
            },
            "visualization": {
                "matplotlib": {
                    "font.family": "serif",
                    "font.serif": ["Latin Modern Roman"],
                    "mathtext.fontset": "cm",
                    "axes.titlesize": 12,
                    "axes.labelsize": 10,
                    "xtick.labelsize": 9,
                    "ytick.labelsize": 9
                },
                "seaborn_style": "whitegrid",
                "dpi": 300,
                "figure_sizes": {
                    "correlation": [12, 10],
                    "time_series": [12, 8],
                    "distribution": [15, 10],
                    "boxplot": [15, 8]
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        try:
            current = self._config
            for k in key.split('.'):
                current = current[k]
            return current
        except (KeyError, TypeError):
            return default 