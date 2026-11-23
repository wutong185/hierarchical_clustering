# -*- coding: utf-8 -*-
"""
step1_program.py
================
ПУНКТ 1 задания — демонстрация программы иерархической кластеризации.

Требование пункта 1:
    «Разработайте программу, которая выполняет иерархическую кластеризацию
     заданного набора данных. Параметрами программы являются набор данных и
     используемая мера сходства кластеров (Single linkage, Complete linkage,
     Group average или Ward distance). Мера сходства объектов произвольная
     (например, можно фиксировать евклидово расстояние). Программа должна
     выполнять визуализацию полученных результатов в виде дендрограммы.»

Здесь показывается, что программа:
    * принимает параметры: набор данных и меру сходства кластеров;
    * использует евклидово расстояние между объектами;
    * строит все n-1 слияний (полное иерархическое дерево);
    * выдаёт дендрограмму (полную и усечённую для читаемости);
    * позволяет обрезать дерево на k кластеров и посчитать ошибку.

Запуск:
    python step1_program.py
    # или напрямую, меняя параметры:
    python clustering_hier.py --method ward --k 5
    python clustering_hier.py --method single
"""

import os

import numpy as np
import pandas as pd

from clustering_hier import (BASE_DIR, FEATURES, METHOD_NAMES, METHODS,
                             METHOD_SHORT, RESULT_DIR, agglomerative,
                             cluster_sizes, cut_tree, load_customers,
                             save_clusters, sse, standardize)
from dendro import draw_dendrogram

DEFAULT_METHOD = "ward"
K = 5


def main():
    print("=" * 76)
    print("ПУНКТ 1 — ПРОГРАММА ИЕРАРХИЧЕСКОЙ КЛАСТЕРИЗАЦИИ")
    print("=" * 76)

    X_raw, df = load_customers()
    Xz, _, _ = standardize(X_raw)

    print(f"Набор данных : data/customers.csv  ({len(Xz)} объектов)")
    print(f"Признаки ({len(FEATURES)}) : {', '.join(FEATURES)}")
    print(f"Мера расстояния между объектами: евклидово (данные стандартизованы)")
    print(f"Доступные меры сходства кластеров:")
    for m in METHODS:
        print(f"    {METHOD_SHORT[m]:<9} — {METHOD_NAMES[m]}")

    # ---------- основная демонстрация: Ward ----------
    method = DEFAULT_METHOD
    print("\n" + "-" * 76)
    print(f"Демонстрация: мера сходства = {METHOD_NAMES[method]}")
    print("-" * 76)

    res = agglomerative(Xz, method)
    print(f"Построено слияний: {len(res['merges'])} (= n - 1 = {len(Xz) - 1})")
    print(f"Высота корня дерева (последнее слияние): {res['heights'][-1]:.4f}")
    print("\nПервые 8 слияний:")
    print(f"  {'№':>2} | {'узел A':>6} + {'узел B':>6} | {'высота':>9} | размер")
    for idx, (a, b, h, sz) in enumerate(res["merges"][:8]):
        print(f"  {idx:>2} | {a:>6} + {b:>6} | {h:>9.4f} | {sz}")

    labels = cut_tree(res, K)
    err = sse(Xz, labels)
    print(f"\nОбрезка дерева на k = {K} кластеров:")
    print(f"  ошибка кластеризации (SSE) = {err:.4f}")
    print(f"  размеры кластеров          = {cluster_sizes(labels)}")

    out_csv = save_clusters(df, labels,
                            os.path.join(RESULT_DIR, f"clusters_{method}_k{K}.csv"))
    print(f"  сохранено: {os.path.relpath(out_csv, BASE_DIR)}")

    # ---------- дендрограммы (пункт 1: визуализация) ----------
    print("\nПостроение дендрограмм ...")
    p_full = draw_dendrogram(
        res, labels,
        f"Дендрограмма (полная): {METHOD_SHORT[method]}, {len(Xz)} объектов\n"
        f"цвет ветвей — кластеры при k = {K}",
        "fig1_dendrogram_full.png", figsize=(20, 5.0))
    print(f"  полная:    {os.path.relpath(p_full, BASE_DIR)}")

    p_tr = draw_dendrogram(
        res, labels,
        f"Дендрограмма (последние 20 слияний): {METHOD_SHORT[method]}, k = {K}\n"
        f"в скобках — число объектов в свёрнутом поддереве",
        "fig1_dendrogram_truncated.png", truncate=20, figsize=(8.6, 5.0))
    print(f"  усечённая: {os.path.relpath(p_tr, BASE_DIR)}")

    # ---------- проверка «параметра»: все 4 меры дают разные деревья ----------
    print("\n" + "-" * 76)
    print("Проверка параметра «мера сходства»: результаты при k = %d" % K)
    print("-" * 76)
    print(f"  {'мера':<10} | {'высота корня':>13} | {'SSE':>10} | размеры кластеров")
    rows = []
    for m in METHODS:
        r = agglomerative(Xz, m)
        lab = cut_tree(r, K)
        e = sse(Xz, lab)
        sz = cluster_sizes(lab)
        rows.append({"method": m, "root_height": r["heights"][-1],
                     "sse_k5": e, "sizes": str(sz)})
        print(f"  {METHOD_SHORT[m]:<10} | {r['heights'][-1]:>13.4f} | {e:>10.3f} | {sz}")

    pd.DataFrame(rows).to_csv(
        os.path.join(RESULT_DIR, "method_overview.csv"), index=False)

    print("\n" + "=" * 76)
    print("Пункт 1 выполнен: параметры задаются, дерево строится, дендрограмма")
    print("построена, обрезка на k кластеров и ошибка выдаются.")
    print("=" * 76)


if __name__ == "__main__":
    main()
