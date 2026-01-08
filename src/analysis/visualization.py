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

    def plot_race_bars(
        self,
        df: pd.DataFrame,
        category: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """Plot bar chart of refusal rates by race."""
        import matplotlib.pyplot as plt

        if category:
            df = df[df["category"] == category]
            title = f"Refusal Rate by Race - Category {category} ({self.CATEGORIES.get(category, '')})"
        else:
            title = "Overall Refusal Rate by Race"

        rates = []
        races = []
        for race in self.RACES:
            race_data = df[df["race"] == race]["is_refused"]
            if len(race_data) > 0:
                rates.append(race_data.mean())
                races.append(self.RACE_ABBREV[race])

        plt.figure(figsize=(12, 6))
        bars = plt.bar(races, rates, color='steelblue')

        # Color highest and lowest differently
        max_idx = rates.index(max(rates))
        min_idx = rates.index(min(rates))
        bars[max_idx].set_color('red')
        bars[min_idx].set_color('green')

        plt.xlabel("Race")
        plt.ylabel("Refusal Rate")
        plt.title(title)
        plt.ylim(0, max(rates) * 1.2 if rates else 1)

        # Add value labels on bars
        for i, (race, rate) in enumerate(zip(races, rates)):
            plt.text(i, rate + 0.01, f"{rate:.1%}", ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            filename = f"race_bars_cat{category}.png" if category else "race_bars_overall.png"
            plt.savefig(self.output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_interaction_heatmap(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None
    ):
        """Plot interaction heatmap showing Race × Category effects."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Compute refusal rates
        pivot = df.pivot_table(
            values="is_refused",
            index="race",
            columns="category",
            aggfunc="mean"
        )

        pivot = pivot.reindex(self.RACES)

        # Compute z-scores for each category to show relative differences
        z_pivot = (pivot - pivot.mean()) / pivot.std()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Raw rates
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1%",
            cmap="RdYlGn_r",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Refusal Rate"},
            ax=ax1
        )
        ax1.set_title("Refusal Rates (Raw)")
        ax1.set_xlabel("Category")
        ax1.set_ylabel("Race")

        # Z-scored
        sns.heatmap(
            z_pivot,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            cbar_kws={"label": "Z-Score"},
            ax=ax2
        )
        ax2.set_title("Refusal Rates (Z-Scored)")
        ax2.set_xlabel("Category")
        ax2.set_ylabel("Race")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "interaction_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_odds_ratios(
        self,
        odds_ratios: list[dict],
        reference_race: str = "White",
        save_path: Optional[str] = None
    ):
        """Plot odds ratios with confidence intervals."""
        import matplotlib.pyplot as plt

        races = [or_result["race1"] for or_result in odds_ratios]
        ors = [or_result["odds_ratio"] for or_result in odds_ratios]
        ci_lowers = [or_result["ci_lower"] for or_result in odds_ratios]
        ci_uppers = [or_result["ci_upper"] for or_result in odds_ratios]

        # Filter out None values
        valid_data = [(r, o, l, u) for r, o, l, u in zip(races, ors, ci_lowers, ci_uppers)
                      if o is not None and l is not None and u is not None]

        if not valid_data:
            return

        races, ors, ci_lowers, ci_uppers = zip(*valid_data)

        # Calculate error bars
        yerr_lower = [o - l for o, l in zip(ors, ci_lowers)]
        yerr_upper = [u - o for o, u in zip(ors, ci_uppers)]

        plt.figure(figsize=(12, 6))

        # Plot on log scale
        y_pos = np.arange(len(races))
        plt.errorbar(ors, y_pos, xerr=[yerr_lower, yerr_upper],
                    fmt='o', markersize=8, capsize=5, capthick=2)

        # Add reference line at OR=1
        plt.axvline(x=1, color='red', linestyle='--', label='No difference (OR=1)')

        plt.yticks(y_pos, races)
        plt.xlabel(f"Odds Ratio (vs {reference_race})")
        plt.ylabel("Race")
        plt.title(f"Refusal Odds Ratios Relative to {reference_race}")
        plt.xscale('log')
        plt.grid(axis='x', alpha=0.3)
        plt.legend()
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "odds_ratios.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_effect_sizes(
        self,
        pairwise_results: list,
        save_path: Optional[str] = None
    ):
        """Plot effect sizes (Cohen's d) for pairwise comparisons."""
        import matplotlib.pyplot as plt

        # Extract significant comparisons
        significant = [r for r in pairwise_results if r.is_significant]

        if not significant:
            print("No significant pairwise differences to plot")
            return

        labels = [f"{r.group1} vs {r.group2}" for r in significant]
        diffs = [r.mean_diff for r in significant]
        ci_lowers = [r.ci_lower for r in significant]
        ci_uppers = [r.ci_upper for r in significant]

        yerr_lower = [d - l for d, l in zip(diffs, ci_lowers)]
        yerr_upper = [u - d for d, u in zip(diffs, ci_uppers)]

        plt.figure(figsize=(12, len(significant) * 0.5 + 2))

        y_pos = np.arange(len(labels))
        colors = ['red' if d > 0 else 'blue' for d in diffs]

        plt.barh(y_pos, diffs, color=colors, alpha=0.6)
        plt.errorbar(diffs, y_pos, xerr=[yerr_lower, yerr_upper],
                    fmt='none', ecolor='black', capsize=3)

        plt.yticks(y_pos, labels)
        plt.xlabel("Mean Difference in Refusal Rate")
        plt.title("Significant Pairwise Differences (with 95% CI)")
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.savefig(self.output_dir / "effect_sizes.png", dpi=300, bbox_inches="tight")
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

    def generate_disparity_latex_table(
        self,
        df: pd.DataFrame,
        caption: str = "Disparity Analysis by Category"
    ) -> str:
        """Generate LaTeX table for disparity metrics."""
        disparities = []
        for cat in ["A", "B", "C", "D", "E"]:
            cat_df = df[df["category"] == cat]
            if len(cat_df) == 0:
                continue

            rates = cat_df.groupby("race")["is_refused"].mean()
            max_race = rates.idxmax()
            min_race = rates.idxmin()
            disparity = rates.max() - rates.min()

            disparities.append({
                "category": cat,
                "max_race": self.RACE_ABBREV.get(max_race, max_race),
                "max_rate": rates.max(),
                "min_race": self.RACE_ABBREV.get(min_race, min_race),
                "min_rate": rates.min(),
                "disparity": disparity
            })

        latex = "\\begin{table}[htbp]\n\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += "\\begin{tabular}{lccccc}\n"
        latex += "\\toprule\n"
        latex += "Category & Max Race & Max Rate & Min Race & Min Rate & Disparity \\\\\n"
        latex += "\\midrule\n"

        for d in disparities:
            cat_name = self.CATEGORIES.get(d["category"], d["category"])
            latex += f"{cat_name} & {d['max_race']} & {d['max_rate']:.1%} & "
            latex += f"{d['min_race']} & {d['min_rate']:.1%} & {d['disparity']:.1%} \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}"

        return latex
