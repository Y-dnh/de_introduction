from typing import Dict, Any, Optional
import yaml
import json
from pathlib import Path


class BaseConfig:
    """Base configuration class for COCO EDA package."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration with default values."""
        self.config = self._get_default_config()
        if config_path:
            self.load_config(config_path)

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            'data': {
                'coco_json_path': '',
                'images_dir': '',
                'classes_of_interest': ['person', 'car', 'pet']
            },
            'filters': {
                'box_area': {
                    'min_percentage': 0.01,  # 1% of image area
                    'max_percentage': 0.80  # 80% of image area
                },
                'mask_ratio': {
                    'min_ratio': 0.1,  # 10% mask-to-box ratio
                    'max_ratio': 1.0  # 100% mask-to-box ratio
                }
            },
            'visualization': {
                'figure_size': (12, 8),
                'dpi': 100,
                'color_palette': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'],
                'max_images_display': 10,
                'histogram_bins': 30
            },
            'analysis': {
                'min_objects_per_image': 1,
                'max_objects_per_image': 50,
                'aspect_ratio_bins': 20
            },
            'export': {
                'output_dir': 'output',
                'save_plots': True,
                'plot_format': 'png',
                'report_format': 'html'
            }
        }

    def load_config(self, config_path: str) -> None:
        """Load configuration from file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                user_config = json.load(f)
        else:
            raise ValueError("Configuration file must be YAML or JSON format")

        self._update_config(user_config)

    def _update_config(self, user_config: Dict[str, Any]) -> None:
        """Recursively update configuration with user values."""

        def update_nested_dict(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    update_nested_dict(base_dict[key], value)
                else:
                    base_dict[key] = value

        update_nested_dict(self.config, user_config)

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config_dict = self.config

        for key in keys[:-1]:
            if key not in config_dict:
                config_dict[key] = {}
            config_dict = config_dict[key]

        config_dict[keys[-1]] = value