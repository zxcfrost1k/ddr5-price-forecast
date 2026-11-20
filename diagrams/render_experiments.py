"""Render three figures from the saved time-based regression experiments."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager, pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports"
FIGURES = ROOT / "diagrams"
FONT = Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf")
if FONT.is_file():
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False, "savefig.dpi": 260})
labels = {"baseline": "Текущая цена", "lasso": "Lasso", "ridge": "Ridge",
          "knn": "Ближайшие соседи", "tree": "Дерево решений",
          "forest": "Случайный лес", "boost": "Градиентный бустинг"}
format_rubles = FuncFormatter(lambda value, _: f"{value:,.0f}".replace(",", " "))


def save(fig, filename):
    fig.savefig(FIGURES / filename, bbox_inches="tight", pad_inches=.09)
    plt.close(fig)


def regularization(summary, baseline):
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.25), layout="constrained")
    subset = summary[summary["group"] == "regularization"].copy()
    subset["alpha"] = subset["parameters"].map(lambda p: json.loads(p)["alpha"])
    for key, color, mark, linestyle in [("lasso", "#111111", "o", "-"), ("ridge", "#6a6a6a", "s", "--")]:
        group = subset[subset["algorithm"] == key].sort_values("alpha")
        for ax, field in zip(axes, ("mae", "nonzero_coefficients")):
            ax.plot(group["alpha"], group[field], color=color, marker=mark, ls=linestyle, label=labels[key])
            ax.set_xscale("log")
            ax.set_xlabel("Сила регуляризации α")
            ax.grid(axis="y", color="#dfdfdf")
    axes[0].axhline(baseline, color="#999999", ls=":", label="Текущая цена")
    axes[0].set_ylabel("MAE проверки, руб.")
    axes[0].yaxis.set_major_formatter(format_rubles)
    axes[1].set_ylabel("Среднее число ненулевых коэффициентов")
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="center right")
    save(fig, "regression_regularization.png")


def depth(summary):
    rows = summary[summary["group"] == "depth"]
    depths = [str(json.loads(p)["max_depth"] or "Без ограничения") for p in rows["parameters"]]
    fig, ax = plt.subplots(figsize=(8.6, 3.3), layout="constrained")
    ax.plot(depths, rows["train_mae"], marker="s", ls="--", color="#686868", label="Обучение")
    ax.plot(depths, rows["mae"], marker="o", color="#171717", label="Проверка")
    ax.set_xlabel("Максимальная глубина дерева")
    ax.set_ylabel("Средняя MAE, руб.")
    ax.yaxis.set_major_formatter(format_rubles)
    ax.grid(axis="y", color="#dfdfdf")
    ax.legend()
    save(fig, "regression_tree_depth.png")


def comparison(by_id, selection):
    final = by_id.loc[[selection["baseline"], *selection["finalists"]]].sort_values("mae").iloc[::-1]
    names = [labels[row.algorithm] + (" (подбор)" if row.group == "grid" else "") for row in final.itertuples()]
    fig, ax = plt.subplots(figsize=(8.9, 4.15), layout="constrained")
    ax.errorbar(final["mae"], names, xerr=final["mae_std"], fmt="o", color="#1c1c1c", ecolor="#999999", capsize=3)
    ax.axvline(by_id.loc["baseline", "mae"], color="#5e5e5e", ls=":", label="Средняя MAE базового правила")
    win = by_id.loc[selection["selected"]]
    winner_label = labels[win.algorithm] + (" (подбор)" if win.group == "grid" else "")
    ax.scatter(win.mae, winner_label, marker="D", color="#111111", zorder=3)
    ax.xaxis.set_major_formatter(format_rubles)
    ax.set_xlabel("MAE по пяти блокам, руб. (среднее ± стандартное отклонение)")
    ax.set_ylim(-1.2, len(final) - .1)
    ax.grid(axis="x", color="#e5e5e5")
    ax.legend(loc="lower right")
    save(fig, "regression_comparison.png")


if __name__ == "__main__":
    data = pd.read_csv(REPORT / "experiment_summary.csv")
    summary = data.set_index("experiment")
    choice = json.loads((REPORT / "model_selection.json").read_text(encoding="utf-8"))
    regularization(data, summary.loc["baseline", "mae"])
    depth(data)
    comparison(summary, choice)
