from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class BBoxConfig:  # Configuration for filtering bounding boxes based on relative size (to image area)
    min_pct = [0.001, 0.1, 0.8]
    max_pct = [0.005, 0.8, 1]
    min_pct = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, ]
    max_pct = [0.001, 0.0025, 0.01, 0.05, 0.1, 0.7, ]


@dataclass
class VisualizationConfig:
    n_examples: int = 10
    histogram_bins: int = 20
    figure_size: tuple[int, int] = (12, 8)
    dpi: int = 100
    color_palette: list[str] = field(default_factory=lambda: ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])

@dataclass
class Config:
    dataset_dir: Path = Path("C:/Users/user/Desktop/coco/")
    output_dir: Path = Path("output")

    classes: tuple[str] = ("Person", "Pet", "Car")

    bbox: BBoxConfig = BBoxConfig()
    visualization: VisualizationConfig = VisualizationConfig()

def create_config():
    return Config()

if __name__ == "__main__":
    from pprint import pprint
    pprint(dir(create_config()))