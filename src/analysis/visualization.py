"""
Visualization for I2I Refusal Bias Results
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


class ResultsVisualizer:
    """Generate visualizations for bias analysis results."""

    RACES = ["White", "Black", "East Asian", "Southeast Asian",
             "Indian", "Middle Eastern", "Latino_Hispanic"]

    RACE_ABBREV = {
        "White": "W",
        "Black": "B",
        "East Asian": "EA",
        "Southeast Asian": "SEA",
        "Indian": "I",
        "Middle Eastern": "ME",
        "Latino_Hispanic": "L"
    }

    CATEGORIES = {
        "A": "Neutral",
        "B": "Occupation",
        "C": "Cultural",
        "D": "Vulnerability",
        "E": "Harmful"
    }

    def __init__(self, output_dir: str = "results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_refusal_heatmap(
        self,
        df: pd.DataFrame,
        title: str = "Refusal Rate by Race and Category",
        save_path: Optional[str] = None
    ):
        """Plot heatmap of refusal rates."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Compute rates
        pivot = df.pivot_table(
            values="is_refused",
            index="race",
            columns="category",
            aggfunc="mean"
        )

        # Reorder
        pivot = pivot.reindex(self.RACES)
        pivot.columns = [self.CATEGORIES.get(c, c) for c in pivot.columns]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2%",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Refusal Rate"}
        )
        plt.title(title)
        plt.xlabel("Prompt Category")
        plt.ylabel("Race")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "refusal_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_disparity_bars(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot bar chart of disparity by category."""
        import matplotlib.pyplot as plt

        disparities = []
        for cat in ["A", "B", "C", "D", "E"]:
            cat_df = df[df["category"] == cat]
            rates = cat_df.groupby("race")["is_refused"].mean()
            disparity = rates.max() - rates.min()
            disparities.append(disparity)

        categories = [self.CATEGORIES[c] for c in ["A", "B", "C", "D", "E"]]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(categories, disparities, color=["green", "orange", "purple", "blue", "red"])

        # Add threshold line
        plt.axhline(y=0.03, color="gray", linestyle="--", label="3pp threshold")

        plt.xlabel("Prompt Category")
        plt.ylabel("Disparity (Max - Min Rate)")
        plt.title("Refusal Rate Disparity Across Races by Category")
        plt.legend()

        # Color bars based on threshold
        for bar, disp in zip(bars, disparities):
            if disp > 0.03:
                bar.set_color("red")
            else:
                bar.set_color("green")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "disparity_bars.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_scs_scores(
        self,
        scs_results: dict,
        save_path: Optional[str] = None
    ):
        """Plot Stereotype Congruence Scores."""
        import matplotlib.pyplot as plt

        prompts = list(scs_results["per_prompt"].keys())
        scores = [scs_results["per_prompt"][p]["scs"] for p in prompts]

        plt.figure(figsize=(12, 6))
        colors = ["red" if s > 0 else "blue" for s in scores]
        plt.bar(prompts, scores, color=colors)

        plt.axhline(y=0, color="black", linewidth=0.5)
        plt.axhline(y=0.1, color="red", linestyle="--", alpha=0.5, label="Strong gatekeeping threshold")
        plt.axhline(y=-0.1, color="blue", linestyle="--", alpha=0.5, label="Strong inverse threshold")

        plt.xlabel("Cultural Prompt ID")
        plt.ylabel("Stereotype Congruence Score")
        plt.title(f"SCS by Prompt (Overall: {scs_results['overall_scs']:.3f})")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "scs_scores.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_model_comparison(
        self,
        results_by_model: dict[str, pd.DataFrame],
        save_path: Optional[str] = None
    ):
        """Compare refusal patterns across models."""
        import matplotlib.pyplot as plt

        models = list(results_by_model.keys())
        x = np.arange(len(self.RACES))
        width = 0.25

        fig, ax = plt.subplots(figsize=(14, 6))

        for i, (model, df) in enumerate(results_by_model.items()):
            rates = [df[df["race"] == race]["is_refused"].mean() for race in self.RACES]
            ax.bar(x + i*width, rates, width, label=model)

        ax.set_xlabel("Race")
        ax.set_ylabel("Refusal Rate")
        ax.set_title("Refusal Rate by Race Across Models")
        ax.set_xticks(x + width)
        ax.set_xticklabels([self.RACE_ABBREV[r] for r in self.RACES])
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    def generate_latex_table(
        self,
        df: pd.DataFrame,
        caption: str = "Refusal Rates by Race and Category"
    ) -> str:
        """Generate LaTeX table for paper."""
        pivot = df.pivot_table(
            values="is_refused",
            index="race",
            columns="category",
            aggfunc="mean"
        )

        pivot = pivot.reindex(self.RACES)

        latex = "\\begin{table}[htbp]\n\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += "\\begin{tabular}{l" + "c" * len(pivot.columns) + "}\n"
        latex += "\\toprule\n"
        latex += "Race & " + " & ".join([self.CATEGORIES.get(c, c) for c in pivot.columns]) + " \\\\\n"
        latex += "\\midrule\n"

        for race in self.RACES:
            row = [self.RACE_ABBREV[race]]
            for cat in pivot.columns:
                val = pivot.loc[race, cat]
                row.append(f"{val:.1%}")
            latex += " & ".join(row) + " \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}"

        return latex
