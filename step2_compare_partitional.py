# -*- coding: utf-8 -*-
"""
step2_compare_partitional.py
============================
ПУНКТ 2 задания — эксперименты на наборе данных из задания «Разделительная
кластеризация» и СРАВНЕНИЕ с результатами того задания.

Что делает:
    * строит иерархическую кластеризацию всеми четырьмя мерами сходства
      и обрезает дерево на k = 3…9 кластеров;
    * для каждого k считает SSE — ту же ошибку, что минимизирует k-Means,
      поэтому результаты сравнимы напрямую;
    * берёт из предыдущего задания ошибки k-Means и k-Medoids
      (data/partitional_baseline.csv) и проверяет, что перенесённый код
      воспроизводит те же значения;
    * строит сравнительные таблицы и рисунки.

Запуск:
    python step2_compare_partitional.py
"""

import os

import numpy as np
import pandas as pd

from clustering_hier import (BASE_DIR, DATA_DIR, METHODS, METHOD_SHORT,
                             RESULT_DIR, agglomerative, cluster_sizes,
                             cut_tree, load_customers, sse, standardize)
from partitional_reference import kmeans, kmedoids
from viz import plot_bars, plot_lines, scatter_panels

K_RANGE = list(range(3, 10))
SEED = 42


def main():
    print("=" * 78)
    print("ПУНКТ 2 — СРАВНЕНИЕ С ЗАДАНИЕМ «РАЗДЕЛИТЕЛЬНАЯ КЛАСТЕРИЗАЦИЯ»")
    print("=" * 78)

    X_raw, df = load_customers()
    Xz, _, _ = standardize(X_raw)
    n = len(Xz)
    print(f"Набор данных: customers.csv ({n} объектов, 7 признаков, "
          f"стандартизация z-score)")
    print(f"Метрика сравнения: SSE (сумма квадратов расстояний до центроидов) — "
          f"именно её минимизирует k-Means\n")

    # ---------- 1) иерархическая: SSE для каждой меры и каждого k ----------
    print("Иерархическая кластеризация (4 меры сходства), SSE при k = 3…9:")
    hier_rows = []
    trees = {}
    for m in METHODS:
        r = agglomerative(Xz, m)
        trees[m] = r
        for k in K_RANGE:
            lab = cut_tree(r, k)
            hier_rows.append({"method": m, "k": k, "sse": sse(Xz, lab),
                              "sizes": str(cluster_sizes(lab))})
    hier = pd.DataFrame(hier_rows)

    piv = hier.pivot(index="k", columns="method", values="sse")
    print(f"\n  {'k':>2} | " + " | ".join(f"{METHOD_SHORT[m]:>9}" for m in METHODS))
    print("  " + "-" * 56)
    for k in K_RANGE:
        print(f"  {k:>2} | " + " | ".join(f"{piv.loc[k, m]:>9.2f}" for m in METHODS))

    hier.to_csv(os.path.join(RESULT_DIR, "hier_error_vs_k.csv"),
                index=False, float_format="%.4f")

    # ---------- 2) эталон из предыдущего задания ----------
    base_path = os.path.join(DATA_DIR, "partitional_baseline.csv")
    base = pd.read_csv(base_path)
    print(f"\nДанные предыдущего задания: {os.path.relpath(base_path, BASE_DIR)}")

    # самопроверка: воспроизводим ли мы цифры предыдущего задания
    print("Самопроверка (перенесённый код vs сохранённые значения):")
    max_diff_km, max_diff_kd = 0.0, 0.0
    for _, row in base.iterrows():
        k = int(row["k"])
        rk = kmeans(Xz, k, seed=SEED)
        rm = kmedoids(Xz, k, seed=SEED)
        max_diff_km = max(max_diff_km, abs(rk["error"] - row["kmeans_error"]))
        max_diff_kd = max(max_diff_kd, abs(rm["error"] - row["kmedoids_error"]))
    print(f"  максимальное расхождение k-Means:   {max_diff_km:.6f}")
    print(f"  максимальное расхождение k-Medoids: {max_diff_kd:.6f}")
    ok = max_diff_km < 1e-3 and max_diff_kd < 1e-3
    print(f"  -> {'СОВПАДАЕТ, сравнение корректно' if ok else 'ЕСТЬ РАСХОЖДЕНИЕ!'}")

    # ---------- 3) сравнительная таблица ----------
    comp = piv.copy().reset_index()
    comp["kmeans_sse"] = base["kmeans_error"].to_numpy()
    comp["kmedoids_err"] = base["kmedoids_error"].to_numpy()
    comp["best_hier"] = comp[[m for m in METHODS]].min(axis=1)
    comp["best_hier_method"] = comp[[m for m in METHODS]].idxmin(axis=1)
    comp.to_csv(os.path.join(RESULT_DIR, "compare_with_partitional.csv"),
                index=False, float_format="%.4f")

    print(f"\n{'k':>2} | {'k-Means':>9} | {'Ward':>9} | {'лучшая иерарх.':>15} | "
          f"{'Ward - kMeans':>14}")
    print("-" * 70)
    for _, row in comp.iterrows():
        k = int(row["k"])
        print(f"{k:>2} | {row['kmeans_sse']:>9.2f} | {row['ward']:>9.2f} | "
              f"{row['best_hier']:>8.2f} ({METHOD_SHORT[row['best_hier_method']]:>8}) | "
              f"{row['ward'] - row['kmeans_sse']:>+14.2f}")

    # ---------- 4) рисунки ----------
    print("\nПостроение рисунков ...")

    series = [{"label": "k-Means", "x": comp["k"], "y": comp["kmeans_sse"],
               "color": "#2c3e50", "marker": "X", "ls": "--"}]
    colors = {"single": "#e74c3c", "complete": "#2980b9",
              "average": "#27ae60", "ward": "#8e44ad"}
    markers = {"single": "o", "complete": "s", "average": "^", "ward": "D"}
    for m in METHODS:
        series.append({"label": f"Иерарх. {METHOD_SHORT[m]}", "x": comp["k"],
                       "y": comp[m], "color": colors[m], "marker": markers[m]})
    p1 = plot_lines(
        series,
        "SSE в зависимости от числа кластеров: иерархическая vs k-Means\n"
        "(набор customers, стандартизованные признаки)",
        "fig2_sse_vs_k.png",
        "Число кластеров k", "SSE")
    print(f"  {os.path.relpath(p1, BASE_DIR)}")

    groups = [str(k) for k in K_RANGE]
    values = [[comp["kmeans_sse"].iloc[i]] + [comp[m].iloc[i] for m in METHODS]
              for i in range(len(comp))]
    # столбцы по алгоритмам, а не по k
    vals_by_algo = [comp["kmeans_sse"].tolist()] + [comp[m].tolist() for m in METHODS]
    labels_by_algo = ["k-Means"] + [METHOD_SHORT[m] for m in METHODS]
    p2 = plot_bars(groups, vals_by_algo, labels_by_algo,
                   "SSE при разных k: k-Means и четыре меры сходства",
                   "fig2_sse_bars.png", "SSE")
    print(f"  {os.path.relpath(p2, BASE_DIR)}")

    # точечные графики: Ward vs k-Means при k = 5
    K_VIS = 5
    lab_ward = cut_tree(trees["ward"], K_VIS)
    rk = kmeans(Xz, K_VIS, seed=SEED)
    panels = [
        {"X": Xz, "labels": lab_ward,
         "subtitle": f"Иерархическая (Ward), k={K_VIS}\n"
                     f"SSE = {sse(Xz, lab_ward):.2f}, размеры {cluster_sizes(lab_ward)}"},
        {"X": Xz, "labels": rk["labels"],
         "subtitle": f"k-Means, k={K_VIS}\n"
                     f"SSE = {rk['error']:.2f}, размеры {cluster_sizes(rk['labels'])}"},
    ]
    p3 = scatter_panels(panels, "Сравнение результатов при k = 5 (проекция PCA)",
                        "fig2_scatter_ward_vs_kmeans.png", ncols=2)
    print(f"  {os.path.relpath(p3, BASE_DIR)}")

    print("\n" + "=" * 78)
    print("Пункт 2 выполнен: сравнение с результатами задания «Разделительная")
    print("кластеризация» проведено по единой метрике SSE.")
    print("=" * 78)


if __name__ == "__main__":
    main()
