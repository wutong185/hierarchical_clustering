# -*- coding: utf-8 -*-
"""
clustering_hier.py
==================
Ядро программы ИЕРАРХИЧЕСКОЙ (агломеративной) кластеризации — реализовано «с нуля»
на numpy, без scipy.cluster.hierarchy.

Задание «Иерархическая кластеризация»:
    п.1 — программа с параметрами «набор данных» и «мера сходства кластеров»
          (Single linkage, Complete linkage, Group average, Ward distance);
          мера сходства объектов — евклидово расстояние; результат визуализируется
          в виде дендрограммы.
    п.2 — эксперименты на наборе данных из задания «Разделительная кластеризация»
          + сравнение с результатами того задания.
    п.3 — эксперименты с разными мерами сходства и их сравнение.

Реализовано:
    * попарные евклидовы расстояния;
    * agglomerative-алгоритм с обновлением расстояний по формуле Ланса–Уильямса
      для четырёх мер сходства;
    * обрезка дерева (получение k кластеров) и расчёт SSE для сравнения
      с k-Means / k-Medoids;
    * история слияний (linkage matrix) для построения дендрограммы.

Запуск (пример, п.1):
    python clustering_hier.py --method ward --k 5
    python clustering_hier.py --method single
"""

import argparse
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULT_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# Признаки для кластеризации (те же, что в задании «Разделительная кластеризация»):
#   Row, CustomerId — идентификаторы;
#   Defaulted       — исключён (150 пропусков из 850).
FEATURES = ["Age", "Education", "YearsEmployed", "Income",
            "CardDebt", "OtherDebt", "DebtIncomeRatio"]

METHODS = ["single", "complete", "average", "ward"]
METHOD_NAMES = {
    "single": "Single linkage (ближайшего соседа)",
    "complete": "Complete linkage (дальнего соседа)",
    "average": "Group average (средней связи)",
    "ward": "Ward distance (метод Уорда)",
}
METHOD_SHORT = {
    "single": "Single",
    "complete": "Complete",
    "average": "Average",
    "ward": "Ward",
}


# =============================================================================
#  Данные
# =============================================================================
def load_customers(path=None, features=None):
    """Читает customers.csv -> (X, df)."""
    path = path or os.path.join(DATA_DIR, "customers.csv")
    features = features or FEATURES
    df = pd.read_csv(path)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"В наборе нет признаков: {missing}")
    X = df[features].to_numpy(dtype=float)
    if np.isnan(X).any():
        raise ValueError("В выбранных признаках есть пропуски")
    return X, df


def standardize(X):
    """Z-стандартизация (как в предыдущем задании)."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def pairwise_sq_distances(X):
    """Матрица КВАДРАТОВ евклидовых расстояний (n, n)."""
    sq = (X ** 2).sum(axis=1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.maximum(D2, 0.0, out=D2)
    np.fill_diagonal(D2, 0.0)
    return D2


# =============================================================================
#  Агломеративная кластеризация (формула Ланса–Уильямса)
# =============================================================================
def agglomerative(X, method="ward"):
    """
    Агломеративная иерархическая кластеризация.

    :param X:      матрица (n, d)
    :param method: 'single' | 'complete' | 'average' | 'ward'
    :return: dict(merges, heights, n)
             merges  — список (child1, child2, height, size) в порядке слияний;
                       листья имеют номера 0..n-1, внутренние узлы n, n+1, ...
             heights — высоты всех n-1 слияний (расстояние, на котором слились)
    """
    if method not in METHODS:
        raise ValueError(f"Неизвестная мера сходства: {method}")

    n = X.shape[0]
    D2 = pairwise_sq_distances(X)          # работаем с квадратами расстояний
    D = np.sqrt(D2)                        # для наглядных высот (single/complete/average)

    # Ward определён через квадраты расстояний, остальные — через сами расстояния
    if method == "ward":
        W = D2.copy()
    else:
        W = D.copy()

    np.fill_diagonal(W, np.inf)

    active = np.ones(n, dtype=bool)
    size = np.ones(n, dtype=int)
    node_id = np.arange(n)                 # номер узла дерева для активного кластера

    merges = []
    heights = []

    for step in range(n - 1):
        # ---- ищем пару активных кластеров с минимальным расстоянием ----
        idx = int(np.argmin(W))
        i, j = divmod(idx, n)

        d_val = float(W[i, j])
        height = np.sqrt(d_val) if method == "ward" else d_val

        ni, nj = int(size[i]), int(size[j])
        new_size = ni + nj
        new_node = n + step

        merges.append((int(node_id[i]), int(node_id[j]), float(height), new_size))
        heights.append(float(height))

        # ---- обновление расстояний (Ланс–Уильямс) ----
        act = np.where(active)[0]
        act = act[(act != i) & (act != j)]
        if len(act):
            ki = W[i, act]
            kj = W[j, act]
            nk = size[act].astype(float)

            if method == "single":
                new_d = np.minimum(ki, kj)
            elif method == "complete":
                new_d = np.maximum(ki, kj)
            elif method == "average":
                new_d = (ni * ki + nj * kj) / float(ni + nj)
            else:  # ward
                denom = (nk + ni + nj)
                new_d = ((nk + ni) * ki + (nk + nj) * kj - nk * d_val) / denom
                np.maximum(new_d, 0.0, out=new_d)

            W[i, act] = new_d
            W[act, i] = new_d

        # ---- деактивируем j ----
        W[j, :] = np.inf
        W[:, j] = np.inf
        W[i, i] = np.inf
        active[j] = False
        size[i] = new_size
        node_id[i] = new_node

    return {"merges": merges, "heights": heights, "n": n, "method": method}


# =============================================================================
#  Обрезка дерева: получение k кластеров
# =============================================================================
def cut_tree(result, k):
    """
    Обрезает дендрограмму так, чтобы получилось ровно k кластеров
    (останавливаем алгоритм после n-k слияний).
    Возвращает массив меток длины n (кластеры пронумерованы 0..k-1).
    """
    n = result["n"]
    if not (1 <= k <= n):
        raise ValueError("k должно быть в диапазоне 1..n")

    parent = list(range(n))
    internal = []                       # внутренние узлы, созданные до остановки
    n_merges_to_do = n - k

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for m in range(n_merges_to_do):
        c1, c2, _, _ = result["merges"][m]
        node = n + m
        parent.append(node)             # новый узел
        r1, r2 = find(c1), find(c2)
        parent[r1] = node
        parent[r2] = node

    # метки: корни оставшихся поддеревьев -> 0..k-1
    roots = {}
    labels = np.empty(n, dtype=int)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels[i] = roots[r]
    return labels


def sse(X, labels):
    """Сумма квадратов расстояний до центроидов кластеров (сравнимо с k-Means)."""
    total = 0.0
    for j in np.unique(labels):
        pts = X[labels == j]
        if len(pts) == 0:
            continue
        c = pts.mean(axis=0)
        total += float(((pts - c) ** 2).sum())
    return total


def cluster_sizes(labels):
    return [int(s) for s in np.bincount(labels)]


def cophenetic_distances(result):
    """
    Копенетические расстояния: для каждой пары объектов — высота слияния,
    на которой они впервые оказались в одном кластере.

    Используются для копенетической корреляции — объективной меры того,
    насколько хорошо дендрограмма сохраняет исходные расстояния.
    """
    n = result["n"]
    cop = np.zeros((n, n), dtype=float)
    leaves = {i: np.array([i]) for i in range(n)}

    for m, (c1, c2, h, _sz) in enumerate(result["merges"]):
        l1, l2 = leaves[c1], leaves[c2]
        if len(l1) and len(l2):
            cop[np.ix_(l1, l2)] = h
            cop[np.ix_(l2, l1)] = h
        node = n + m
        leaves[node] = np.concatenate([l1, l2])
        # освобождаем память: листья детей больше не нужны
        leaves.pop(c1, None)
        leaves.pop(c2, None)

    return cop


def cophenetic_correlation(result, X=None):
    """
    Копенетическая корреляция — коэффициент корреляции Пирсона между
    копенетическими расстояниями и исходными расстояниями между объектами.
    Чем ближе к 1, тем лучше дерево отражает структуру данных.
    """
    n = result["n"]
    if X is None:
        raise ValueError("Нужна матрица X для расчёта исходных расстояний")
    cop = cophenetic_distances(result)
    D = np.sqrt(pairwise_sq_distances(X))

    iu = np.triu_indices(n, k=1)
    a, b = cop[iu], D[iu]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def save_clusters(df, labels, path):
    out = df.copy()
    out["cluster"] = labels
    out.to_csv(path, index=False)
    return path


# =============================================================================
#  CLI (п.1 задания)
# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Иерархическая кластеризация (Single / Complete / Average / Ward)")
    p.add_argument("--data", default=None, help="путь к CSV (по умолчанию data/customers.csv)")
    p.add_argument("--method", default="ward", choices=METHODS,
                   help="мера сходства кластеров")
    p.add_argument("--k", type=int, default=5, help="число кластеров при обрезке дерева")
    p.add_argument("--out", default=None, help="файл для сохранения кластеров")
    args = p.parse_args()

    X_raw, df = load_customers(args.data)
    Xz, _, _ = standardize(X_raw)

    res = agglomerative(Xz, args.method)
    labels = cut_tree(res, args.k)
    err = sse(Xz, labels)

    print("=" * 74)
    print("ИЕРАРХИЧЕСКАЯ КЛАСТЕРИЗАЦИЯ — пункт 1 задания")
    print("=" * 74)
    print(f"Набор данных:        {os.path.basename(args.data or 'customers.csv')}")
    print(f"Объектов:            {len(Xz)}")
    print(f"Признаки ({len(FEATURES)}):        {', '.join(FEATURES)}")
    print(f"Мера сходства:       {METHOD_NAMES[args.method]}")
    print(f"Мера расстояния:     евклидово (данные стандартизованы)")
    print(f"Слияний построено:   {len(res['merges'])} (n-1)")
    print("-" * 74)
    print(f"Высота последнего слияния (корень дерева): {res['heights'][-1]:.4f}")
    print(f"Первые 10 слияний (номер узла, высота, размер):")
    for idx, m in enumerate(res["merges"][:10]):
        print(f"  {idx:>2}: {m[0]:>4} + {m[1]:>4}  на высоте {m[2]:.4f}  ->  "
              f"кластер размера {m[3]}")
    print("-" * 74)
    print(f"Обрезка дерева на k = {args.k} кластеров:")
    print(f"  ошибка кластеризации (SSE) = {err:.4f}")
    print(f"  размеры кластеров          = {cluster_sizes(labels)}")

    out_path = args.out or os.path.join(RESULT_DIR, f"clusters_{args.method}_k{args.k}.csv")
    save_clusters(df, labels, out_path)
    print(f"  координаты и кластеры сохранены: {os.path.relpath(out_path, BASE_DIR)}")
    print("=" * 74)


if __name__ == "__main__":
    main()
