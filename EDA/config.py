from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class BBoxConfig:
    min_pct: float = 0.01
    max_pct: float = 0.80

@dataclass
class MaskBoxRatioConfig:
    min: float = 0.2
    max: float = 1.0

@dataclass
class VisualizationConfig:
    n_examples: int = 5
    histogram_bins: int = 20
    figure_size: tuple[int, int] = (12, 8)
    dpi: int = 100
    color_palette: list[str] = field(default_factory=lambda: ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])

@dataclass
class Config:
    dataset_dir: Path = Path("C:/Users/user/Desktop/pascal_voc/")
    output_dir: Path = Path("output")

    classes: tuple[str] = ("Person", "Pet", "Car")

    bbox: BBoxConfig = BBoxConfig()
    mask_box_ratio: MaskBoxRatioConfig = MaskBoxRatioConfig()
    visualization: VisualizationConfig = VisualizationConfig()

def create_config():
    return Config()

if __name__ == "__main__":
    from pprint import pprint
    pprint(dir(create_config()))