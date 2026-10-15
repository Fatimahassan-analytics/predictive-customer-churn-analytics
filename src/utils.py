"""Shared utilities for the Churn project."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

RANDOM_SEED = 42
plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 100,
    "axes.grid": True,
    "axes.grid.which": "both",
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_style("whitegrid", {"axes.grid": True})


def save_fig(name: str, fig_dir: Path = Path("artefacts")) -> Path:
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / name
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    return path


def save_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def pretty_print_cv(cv_results: dict) -> str:
    lines = []
    for metric, vals in cv_results.items():
        lines.append(f"  {metric:>12}: {vals['mean']:.4f} ± {vals['std']:.4f}")
    return "\n".join(lines)


def plot_churn_distribution(df: pd.DataFrame, target: str = "Churn"):
    ax = sns.countplot(data=df, x=target, palette="pastel")
    ax.set_title("Churn class distribution")
    ax.set_ylabel("Number of customers")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}",
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=10)
    return ax


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list):
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation heatmap (numeric features)")
    return ax


def plot_churn_by_feature(df: pd.DataFrame, feature: str, target: str = "Churn"):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if df[feature].dtype == "object" or df[feature].nunique() < 15:
        sns.countplot(data=df, x=feature, hue=target, palette="Set2", ax=ax)
        ax.set_title(f"Churn by {feature}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    else:
        sns.histplot(data=df, x=feature, hue=target, kde=True, palette="Set2", ax=ax)
        ax.set_title(f"Distribution of {feature} by churn")
    return ax


def plot_confusion(cm: list, labels: list = None):
    labels = labels or ["No churn", "Churn"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    return ax
