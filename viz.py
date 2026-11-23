# -*- coding: utf-8 -*-
"""
viz.py
======
Вспомогательная визуализация для задания «Иерархическая кластеризация»:
    * точечные графики кластеров (проекция на 2 главные компоненты, PCA)
      — для сравнения с результатами задания «Разделительная кластеризация»;
    * графики зависимостей (SSE от k, сравнение мер сходства).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from clustering_hier import BASE_DIR

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

PALETTE = ["#e74c3c", "#2980b9", "#27ae60", "#f39c12", "#8e44ad",
           "#16a085", "#d35400", "#2c3e50", "#c0392b"]
RESULT_DIR = os.path.join(BASE_DIR, "results")


def pca_2d(X):
    """Проекция на первые две главные компоненты."""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T, (S ** 2) / np.sum(S ** 2)


def scatter_clusters(X, labels, title, filename, figsize=(7.2, 5.6),
                     centers=None, dpi=150):
    """Точечный график: цвет точки = кластер."""
    X2, expl = pca_2d(X)
    k = int(labels.max() + 1)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for j in range(k):
        m = labels == j
        ax.scatter(X2[m, 0], X2[m, 1], s=24, alpha=0.75,
                   color=PALETTE[j % len(PALETTE)], linewidths=0,
                   label=f"Кластер {j + 1} ({int(m.sum())})")
    if centers is not None and len(centers) > 0:
        mu = X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
        C2 = (np.asarray(centers) - mu) @ Vt[:2].T
        ax.scatter(C2[:, 0], C2[:, 1], s=180, marker="X", color="black",
                   edgecolors="white", linewidths=1.5, zorder=5)
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel(f"Главная компонента 1 ({expl[0]:.1%})")
    ax.set_ylabel(f"Главная компонента 2 ({expl[1]:.1%})")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    path = os.path.join(RESULT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def scatter_panels(panels, title, filename, ncols=2, figsize=None, dpi=150):
    """Сетка точечных графиков."""
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    if figsize is None:
        figsize = (5.6 * ncols, 4.8 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes = np.atleast_1d(axes).ravel()
    for ax, p in zip(axes, panels):
        X2, _ = pca_2d(p["X"])
        k = int(p["labels"].max() + 1)
        for j in range(k):
            m = p["labels"] == j
            ax.scatter(X2[m, 0], X2[m, 1], s=14, alpha=0.75,
                       color=PALETTE[j % len(PALETTE)], linewidths=0)
        ax.set_title(p["subtitle"], fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(alpha=0.2)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(RESULT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_lines(series, title, filename, xlabel, ylabel,
               figsize=(8.0, 5.0), dpi=150):
    """
    :param series: список dict(label, x, y, color, marker, ls)
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for s in series:
        ax.plot(s["x"], s["y"], marker=s.get("marker", "o"),
                color=s.get("color"), ls=s.get("ls", "-"), lw=2, ms=5.5,
                label=s["label"])
    ax.set_title(title, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(sorted({v for s in series for v in s["x"]}))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    path = os.path.join(RESULT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_bars(groups, values, labels, title, filename, ylabel,
              figsize=(9.0, 4.8), dpi=150):
    """Сгруппированная столбчатая диаграмма."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    n = len(groups)
    width = 0.8 / max(len(values), 1)
    xs = np.arange(n)
    for i, (vals, lab) in enumerate(zip(values, labels)):
        ax.bar(xs + i * width - 0.4 + width / 2, vals, width,
               label=lab, color=PALETTE[i % len(PALETTE)], alpha=0.88)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    path = os.path.join(RESULT_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    return path
