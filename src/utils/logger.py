"""
Experiment Logger
Unified logging for I2I refusal bias experiments
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Any
import sys


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    console_level: str = "INFO",
    file_level: str = "DEBUG"
) -> logging.Logger:
    """Setup a logger with console and file handlers."""

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers = []

    # Format
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, console_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(getattr(logging, file_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ExperimentLogger:
    """
    Structured experiment logger.

    Logs:
    - Experiment start/end
    - Each generation attempt
    - Refusals with details
    - Errors
    - Timing information
    - Summary statistics
    """

    def __init__(
        self,
        experiment_id: str,
        model_name: str,
        log_dir: Path,
        console_level: str = "INFO"
    ):
        self.experiment_id = experiment_id
        self.model_name = model_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup loggers
        self.main_log = self.log_dir / "experiment.log"
        self.refusal_log = self.log_dir / "refusals.jsonl"
        self.error_log = self.log_dir / "errors.jsonl"
        self.timing_log = self.log_dir / "timings.jsonl"

        self.logger = setup_logger(
            f"exp.{model_name}",
            self.main_log,
            console_level=console_level
        )

        # Statistics
        self.stats = {
            "total": 0,
            "success": 0,
            "refused": 0,
            "errors": 0,
            "by_category": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
            "by_race": {},
            "refusals_by_race": {},
            "refusals_by_category": {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0},
        }

        self.start_time = None

    def start_experiment(self, total_requests: int):
        """Log experiment start."""
        self.start_time = datetime.now()
        self.stats["total_expected"] = total_requests

        self.logger.info("=" * 60)
        self.logger.info(f"EXPERIMENT START: {self.experiment_id}")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Total requests: {total_requests}")
        self.logger.info(f"Start time: {self.start_time.isoformat()}")
        self.logger.info("=" * 60)

    def log_generation(
        self,
        prompt_id: str,
        prompt_text: str,
        category: str,
        race: str,
        gender: str,
        age: str,
        success: bool,
        is_refused: bool,
        refusal_type: Optional[str] = None,
        error_message: Optional[str] = None,
        latency_ms: float = 0,
        output_path: Optional[str] = None
    ):
        """Log a single generation attempt."""
        self.stats["total"] += 1
        self.stats["by_category"][category] = self.stats["by_category"].get(category, 0) + 1

        if race not in self.stats["by_race"]:
            self.stats["by_race"][race] = 0
            self.stats["refusals_by_race"][race] = 0
        self.stats["by_race"][race] += 1

        if success and not is_refused:
            self.stats["success"] += 1
            status = "SUCCESS"
        elif is_refused:
            self.stats["refused"] += 1
            self.stats["refusals_by_category"][category] += 1
            self.stats["refusals_by_race"][race] += 1
            status = f"REFUSED ({refusal_type})"

            # Log refusal details
            self._log_refusal({
                "timestamp": datetime.now().isoformat(),
                "prompt_id": prompt_id,
                "prompt_text": prompt_text,
                "category": category,
                "race": race,
                "gender": gender,
                "age": age,
                "refusal_type": refusal_type,
                "error_message": error_message,
            })
        else:
            self.stats["errors"] += 1
            status = f"ERROR"

            # Log error details
            self._log_error({
                "timestamp": datetime.now().isoformat(),
                "prompt_id": prompt_id,
                "race": race,
                "error_message": error_message,
            })

        # Log timing
        self._log_timing({
            "prompt_id": prompt_id,
            "race": race,
            "category": category,
            "latency_ms": latency_ms,
            "status": status,
        })

        # Console log
        progress = f"[{self.stats['total']}/{self.stats.get('total_expected', '?')}]"
        self.logger.info(
            f"{progress} {prompt_id} | {race[:10]:10} | {category} | {status:20} | {latency_ms:.0f}ms"
        )

        if output_path:
            self.logger.debug(f"  -> Saved: {output_path}")

    def _log_refusal(self, data: dict):
        """Append refusal to JSONL log."""
        with open(self.refusal_log, "a") as f:
            f.write(json.dumps(data) + "\n")

    def _log_error(self, data: dict):
        """Append error to JSONL log."""
        with open(self.error_log, "a") as f:
            f.write(json.dumps(data) + "\n")

    def _log_timing(self, data: dict):
        """Append timing to JSONL log."""
        with open(self.timing_log, "a") as f:
            f.write(json.dumps(data) + "\n")

    def log_checkpoint(self, message: str = ""):
        """Log a checkpoint."""
        elapsed = datetime.now() - self.start_time if self.start_time else None
        self.logger.info("-" * 40)
        self.logger.info(f"CHECKPOINT: {message}")
        self.logger.info(f"  Total: {self.stats['total']}")
        self.logger.info(f"  Success: {self.stats['success']}")
        self.logger.info(f"  Refused: {self.stats['refused']}")
        self.logger.info(f"  Errors: {self.stats['errors']}")
        if elapsed:
            self.logger.info(f"  Elapsed: {elapsed}")
        self.logger.info("-" * 40)

    def end_experiment(self) -> dict:
        """Log experiment end and return summary."""
        end_time = datetime.now()
        elapsed = end_time - self.start_time if self.start_time else None

        # Calculate rates
        total = self.stats["total"]
        refusal_rate = self.stats["refused"] / total if total > 0 else 0
        success_rate = self.stats["success"] / total if total > 0 else 0

        summary = {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": end_time.isoformat(),
            "elapsed_seconds": elapsed.total_seconds() if elapsed else None,
            "total_requests": total,
            "success_count": self.stats["success"],
            "refusal_count": self.stats["refused"],
            "error_count": self.stats["errors"],
            "refusal_rate": refusal_rate,
            "success_rate": success_rate,
            "by_category": self.stats["by_category"],
            "by_race": self.stats["by_race"],
            "refusals_by_category": self.stats["refusals_by_category"],
            "refusals_by_race": self.stats["refusals_by_race"],
        }

        # Calculate per-race refusal rates
        summary["refusal_rate_by_race"] = {
            race: self.stats["refusals_by_race"].get(race, 0) / count if count > 0 else 0
            for race, count in self.stats["by_race"].items()
        }

        # Log summary
        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info(f"Total: {total}")
        self.logger.info(f"Success: {self.stats['success']} ({success_rate:.1%})")
        self.logger.info(f"Refused: {self.stats['refused']} ({refusal_rate:.1%})")
        self.logger.info(f"Errors: {self.stats['errors']}")
        self.logger.info(f"Duration: {elapsed}")
        self.logger.info("")
        self.logger.info("Refusal Rate by Race:")
        for race, rate in sorted(summary["refusal_rate_by_race"].items(), key=lambda x: -x[1]):
            self.logger.info(f"  {race:20}: {rate:.1%}")
        self.logger.info("")
        self.logger.info("Refusal Rate by Category:")
        for cat in ["A", "B", "C", "D", "E"]:
            cat_total = self.stats["by_category"].get(cat, 0)
            cat_refused = self.stats["refusals_by_category"].get(cat, 0)
            cat_rate = cat_refused / cat_total if cat_total > 0 else 0
            self.logger.info(f"  Category {cat}: {cat_refused}/{cat_total} ({cat_rate:.1%})")
        self.logger.info("=" * 60)

        # Save summary
        summary_path = self.log_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        return summary
