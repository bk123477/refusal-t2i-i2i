"""
Experiment Configuration
Unified paths, naming conventions, and logging settings
"""

from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json


@dataclass
class PathConfig:
    """Unified path configuration."""

    # Base directories
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    results_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "results")
    logs_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")

    # Source images
    source_images_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "source_images" / "fairface")

    # Prompts
    prompts_file: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "prompts" / "i2i_prompts.json")

    def __post_init__(self):
        """Convert all paths to Path objects and create directories."""
        for attr_name in ['project_root', 'data_dir', 'results_dir', 'logs_dir', 'source_images_dir', 'prompts_file']:
            attr = getattr(self, attr_name)
            if isinstance(attr, str):
                setattr(self, attr_name, Path(attr))

    def setup_experiment_dirs(self, model_name: str, experiment_id: Optional[str] = None) -> dict:
        """
        Setup experiment directories for a specific model run.

        Returns dict with all relevant paths.
        """
        if experiment_id is None:
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        exp_dir = self.results_dir / model_name / experiment_id

        paths = {
            "experiment_dir": exp_dir,
            "images_dir": exp_dir / "images",
            "refusals_dir": exp_dir / "refusals",
            "logs_dir": exp_dir / "logs",
            "results_file": exp_dir / "results.json",
            "summary_file": exp_dir / "summary.json",
            "log_file": exp_dir / "logs" / "experiment.log",
        }

        # Create all directories
        for key, path in paths.items():
            if key.endswith("_dir"):
                path.mkdir(parents=True, exist_ok=True)
            elif key.endswith("_file"):
                path.parent.mkdir(parents=True, exist_ok=True)

        return paths


@dataclass
class NamingConfig:
    """
    Unified naming convention for images and results.

    Format: {model}_{race}_{gender}_{age}_{prompt_id}_{status}.{ext}

    Examples:
        - flux_White_Male_20-29_A01_success.png
        - step1x_Black_Female_30-39_E05_refused.png
        - qwen_East-Asian_Male_40-49_B03_success.png
    """

    @staticmethod
    def get_image_filename(
        model: str,
        race: str,
        gender: str,
        age: str,
        prompt_id: str,
        status: str = "success",
        ext: str = "png"
    ) -> str:
        """Generate standardized image filename."""
        # Sanitize race (replace spaces with hyphens)
        race_clean = race.replace(" ", "-").replace("/", "-")
        return f"{model}_{race_clean}_{gender}_{age}_{prompt_id}_{status}.{ext}"

    @staticmethod
    def parse_image_filename(filename: str) -> dict:
        """Parse image filename back to components."""
        parts = filename.rsplit(".", 1)[0].split("_")
        if len(parts) >= 6:
            return {
                "model": parts[0],
                "race": parts[1].replace("-", " "),
                "gender": parts[2],
                "age": parts[3],
                "prompt_id": parts[4],
                "status": parts[5]
            }
        return {}

    @staticmethod
    def get_source_image_filename(race: str, gender: str, age: str) -> str:
        """Get source image filename."""
        race_clean = race.replace(" ", "_").replace("/", "_")
        return f"{race_clean}_{gender}_{age}.jpg"


@dataclass
class LogConfig:
    """Logging configuration."""

    # Log levels
    console_level: str = "INFO"
    file_level: str = "DEBUG"

    # Log format
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"

    # What to log
    log_prompts: bool = True
    log_timings: bool = True
    log_refusals: bool = True
    log_errors: bool = True


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""

    # Experiment metadata
    experiment_name: str = "i2i_refusal_bias"
    experiment_id: Optional[str] = None

    # Model settings
    model_name: str = "flux"
    device: str = "cuda"

    # Generation settings
    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    seed: int = 42

    # Batch settings
    save_interval: int = 10  # Save results every N images

    # Paths
    paths: PathConfig = field(default_factory=PathConfig)
    naming: NamingConfig = field(default_factory=NamingConfig)
    logging: LogConfig = field(default_factory=LogConfig)

    def __post_init__(self):
        if self.experiment_id is None:
            self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "device": self.device,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "save_interval": self.save_interval,
        }

    def save(self, path: Path):
        """Save config to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


# Model-specific configurations
MODEL_CONFIGS = {
    "flux": {
        "num_inference_steps": 50,
        "guidance_scale": 4.0,
        "use_quantized": False,
        "use_remote_text_encoder": False,
    },
    "step1x": {
        "num_inference_steps": 50,
        "true_cfg_scale": 6.0,
        "enable_thinking": True,
        "enable_reflection": True,
    },
    "qwen": {
        "num_inference_steps": 40,
        "true_cfg_scale": 4.0,
        "guidance_scale": 1.0,
    }
}
