# -*- coding: utf-8 -*-
"""
step3_linkage_compare.py
========================
ПУНКТ 3 задания — эксперименты на выбранном наборе данных с РАЗНЫМИ мерами
сходства кластеров и их СРАВНЕНИЕ между собой.

Сравниваются четыре меры:
    Single linkage, Complete linkage, Group average, Ward distance.

Показатели сравнения:
    * дендрограммы (визуально);
    * SSE после обрезки дерева на k = 3…9;
    * доля самого крупного кластера при k = 5 — количественная мера
      «эффекта цепочки» (chaining), характерного для Single/Average;
    * копенетическая корреляция — насколько дерево сохраняет исходные расстояния.

Запуск:
    python step3_linkage_compare.py
"""

import os

import numpy as np
import pandas as pd

from clustering_hier import (BASE_DIR, METHODS, METHOD_NAMES, METHOD_SHORT,
                             RESULT_DIR, agglomerative, cluster_sizes,
                             cophenetic_correlation, cut_tree,
                             load_customers, sse, standardize)
from dendro import draw_dendrogram_panels
from viz import plot_bars, plot_lines

K_RANGE = list(range(3, 10))
K_VIS = 5


def main():
    print("=" * 78)
    print("ПУНКТ 3 — СРАВНЕНИЕ МЕР СХОДСТВА КЛАСТЕРОВ")
    print("=" * 78)

    X_raw, df = load_customers()
    Xz, _, _ = standardize(X_raw)
    n = len(Xz)
    print(f"Набор данных: customers.csv ({n} объектов, 7 признаков)\n")

    # ---------- строим деревья и собираем показатели ----------
    trees, labels_at_k = {}, {}
    rows = []
    for m in METHODS:
        r = agglomerative(Xz, m)
        trees[m] = r
        lab5 = cut_tree(r, K_VIS)
        labels_at_k[m] = lab5
        sizes5 = cluster_sizes(lab5)
        rows.append({
            "method": m,
            "root_height": r["heights"][-1],
            "sse_k5": sse(Xz, lab5),
            "largest_share": max(sizes5) / n,
            "n_nonempty": sum(1 for s in sizes5 if s > 0),
            "cophenetic": cophenetic_correlation(r, Xz),
            "sizes_k5": str(sizes5),
        })
    summary = pd.DataFrame(rows)

    # ---------- SSE по k для каждой меры ----------
    curve_rows = []
    for m in METHODS:
        for k in K_RANGE:
            lab = cut_tree(trees[m], k)
            curve_rows.append({"method": m, "k": k, "sse": sse(Xz, lab)})
    curves = pd.DataFrame(curve_rows)
    curves.to_csv(os.path.join(RESULT_DIR, "linkage_sse_vs_k.csv"),
                  index=False, float_format="%.4f")

    print(f"Показатели при k = {K_VIS}:")
    print(f"  {'мера':<10} | {'корень':>8} | {'SSE':>9} | {'доля крупн.':>11} | "
          f"{'копенет.корр':>12}")
    print("  " + "-" * 62)
    for _, r in summary.iterrows():
        print(f"  {METHOD_SHORT[r['method']]:<10} | {r['root_height']:>8.2f} | "
              f"{r['sse_k5']:>9.2f} | {r['largest_share']:>10.1%} | "
              f"{r['cophenetic']:>12.3f}")

    print("\nРазмеры кластеров при k = %d:" % K_VIS)
    for _, r in summary.iterrows():
        print(f"  {METHOD_SHORT[r['method']]:<10} : {r['sizes_k5']}")

    summary.to_csv(os.path.join(RESULT_DIR, "linkage_summary.csv"),
                   index=False, float_format="%.4f")

    # ---------- рисунки ----------
    print("\nПостроение рисунков ...")

    p1 = draw_dendrogram_panels(
        [trees[m] for m in METHODS],
        [labels_at_k[m] for m in METHODS],
        "Дендрограммы для четырёх мер сходства (показаны последние 18 слияний)\n"
        "цвет — кластеры при k = 5; в скобках — размеры свёрнутых поддеревьев",
        "fig3_dendrograms.png")
    print(f"  {os.path.relpath(p1, BASE_DIR)}")

    colors = {"single": "#e74c3c", "complete": "#2980b9",
              "average": "#27ae60", "ward": "#8e44ad"}
    markers = {"single": "o", "complete": "s", "average": "^", "ward": "D"}
    series = []
    for m in METHODS:
        sub = curves[curves["method"] == m]
        series.append({"label": MODEL_LABEL(m), "x": sub["k"], "y": sub["sse"],
                       "color": colors[m], "marker": markers[m]})
    p2 = plot_lines(
        series,
        "SSE в зависимости от k для четырёх мер сходства\n(набор customers)",
        "fig3_sse_vs_k.png", "Число кластеров k", "SSE")
    print(f"  {os.path.relpath(p2, BASE_DIR)}")

    p3 = plot_bars(
        [METHOD_SHORT[m] for m in METHODS],
        [summary["largest_share"].tolist()],
        [f"Доля самого крупного кластера (k = {K_VIS})"],
        "«Эффект цепочки»: доля самого крупного кластера при k = 5\n"
        "чем ближе к 100%, тем сильнее выражено вырождение кластеров",
        "fig3_chaining.png", "Доля объектов в крупнейшем кластере")
    print(f"  {os.path.relpath(p3, BASE_DIR)}")

    p4 = plot_bars(
        [METHOD_SHORT[m] for m in METHODS],
        [summary["cophenetic"].tolist()],
        ["Копенетическая корреляция"],
        "Копенетическая корреляция: насколько дендрограмма сохраняет\n"
        "исходные расстояния между объектами (чем ближе к 1, тем лучше)",
        "fig3_cophenetic.png", "Коэффициент корреляции")
    print(f"  {os.path.relpath(p4, BASE_DIR)}")

    print("\n" + "=" * 78)
    print("Пункт 3 выполнен: четыре меры сходства сравнены по дендрограммам,")
    print("SSE, доле крупнейшего кластера и копенетической корреляции.")
    print("=" * 78)


def MODEL_LABEL(m):
    return {"single": "Single linkage",
            "complete": "Complete linkage",
            "average": "Group average",
            "ward": "Ward distance"}[m]


if __name__ == "__main__":
    main()
