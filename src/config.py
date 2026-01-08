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

    # Source images base (versions inside: V1/, V2/, ..., V7/, final/)
    source_images_base: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "source_images" / "fairface")

    # Default version (use 'final' for curated images)
    source_version: str = "final"

    # Prompts
    prompts_file: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data" / "prompts" / "i2i_prompts.json")

    @property
    def source_images_dir(self) -> Path:
        """Get source images directory for current version."""
        return self.source_images_base / self.source_version

    def set_version(self, version: str):
        """Set source image version."""
        self.source_version = version
        return self

    def __post_init__(self):
        """Convert all paths to Path objects and create directories."""
        for attr_name in ['project_root', 'data_dir', 'results_dir', 'logs_dir', 'source_images_base', 'prompts_file']:
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

    Source images: {RaceCode}/{RaceCode}_{Gender}_{AgeCode}.jpg
    Output images: {RaceCode}/{PromptID}_{RaceCode}_{Gender}_{AgeCode}_{status}.png

    Race codes: Black, EastAsian, Indian, Latino, MiddleEastern, SoutheastAsian, White
    Age codes: 20s, 30s, 40s, 50s, 60s, 70plus
    """

    # Race code mapping (FairFace label -> code)
    RACE_TO_CODE = {
        "Black": "Black",
        "East Asian": "EastAsian",
        "Indian": "Indian",
        "Latino_Hispanic": "Latino",
        "Middle Eastern": "MiddleEastern",
        "Southeast Asian": "SoutheastAsian",
        "White": "White"
    }

    CODE_TO_RACE = {v: k for k, v in RACE_TO_CODE.items()}

    # Age code mapping (FairFace label -> code)
    AGE_TO_CODE = {
        "20-29": "20s",
        "30-39": "30s",
        "40-49": "40s",
        "50-59": "50s",
        "60-69": "60s",
        "more than 70": "70plus"
    }

    CODE_TO_AGE = {v: k for k, v in AGE_TO_CODE.items()}

    @classmethod
    def get_race_code(cls, race: str) -> str:
        """Convert FairFace race label to code."""
        return cls.RACE_TO_CODE.get(race, race.replace(" ", "").replace("_", ""))

    @classmethod
    def get_age_code(cls, age: str) -> str:
        """Convert FairFace age label to code."""
        return cls.AGE_TO_CODE.get(age, age)

    @classmethod
    def get_output_filename(
        cls,
        prompt_id: str,
        race_code: str,
        gender: str,
        age_code: str,
        status: str = "success",
        ext: str = "png"
    ) -> str:
        """
        Generate output image filename.
        Format: {PromptID}_{RaceCode}_{Gender}_{AgeCode}_{status}.png
        Example: A01_Black_Male_20s_success.png
        """
        return f"{prompt_id}_{race_code}_{gender}_{age_code}_{status}.{ext}"

    @classmethod
    def get_output_path(
        cls,
        base_dir: Path,
        prompt_id: str,
        race_code: str,
        gender: str,
        age_code: str,
        status: str = "success",
        ext: str = "png"
    ) -> Path:
        """
        Get full output path with race subfolder.
        Structure: {base_dir}/{RaceCode}/{filename}
        """
        filename = cls.get_output_filename(prompt_id, race_code, gender, age_code, status, ext)
        return base_dir / race_code / filename

    @classmethod
    def parse_output_filename(cls, filename: str) -> dict:
        """Parse output image filename back to components."""
        parts = filename.rsplit(".", 1)[0].split("_")
        if len(parts) >= 5:
            return {
                "prompt_id": parts[0],
                "race_code": parts[1],
                "gender": parts[2],
                "age_code": parts[3],
                "status": parts[4]
            }
        return {}

    @classmethod
    def get_source_image_path(cls, base_dir: Path, race_code: str, gender: str, age_code: str) -> Path:
        """
        Get source image path.
        Structure: {base_dir}/{RaceCode}/{RaceCode}_{Gender}_{AgeCode}.jpg
        """
        filename = f"{race_code}_{gender}_{age_code}.jpg"
        return base_dir / race_code / filename


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
