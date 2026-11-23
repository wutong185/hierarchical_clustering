# -*- coding: utf-8 -*-
"""
partitional_reference.py
========================
Эталонная реализация k-Means и k-Medoids из задания «Разделительная кластеризация»
(код перенесён без изменений), чтобы в этом проекте можно было корректно
СРАВНИТЬ иерархическую кластеризацию с результатами предыдущего задания
(пункт 2 задания).

При том же наборе данных, той же стандартизации и том же seed = 42 значения
ошибок должны совпасть с data/partitional_baseline.csv — это используется
как самопроверка.
"""

import numpy as np


def _sq_distances(X, C):
    return ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)


def _distances(X, Y):
    return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))


def kmeans(X, k, seed=42, max_iter=300, tol=1e-8):
    """k-Means (Ллойд) + инициализация k-means++; ошибка = SSE."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]

    centers = [X[rng.randint(n)]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
        total = d2.sum()
        if total <= 0:
            centers.append(X[rng.randint(n)])
            continue
        probs = d2 / total
        centers.append(X[rng.choice(n, p=probs)])
    C = np.array(centers, dtype=float)

    labels = np.zeros(n, dtype=int)
    for it in range(1, max_iter + 1):
        D = _sq_distances(X, C)
        new_labels = D.argmin(axis=1)

        new_C = C.copy()
        for j in range(k):
            members = X[new_labels == j]
            if len(members) > 0:
                new_C[j] = members.mean(axis=0)
            else:
                far = D.min(axis=1).argmax()
                new_C[j] = X[far]

        shift = np.abs(new_C - C).max()
        C = new_C
        if np.array_equal(new_labels, labels) or shift < tol:
            labels = new_labels
            break
        labels = new_labels

    D = _sq_distances(X, C)
    labels = D.argmin(axis=1)
    return {"labels": labels, "centers": C,
            "error": float(D.min(axis=1).sum()), "n_iter": it}


def kmedoids(X, k, seed=42, max_iter=200, refine=3):
    """k-Medoids (PAM: BUILD + чередование + SWAP); ошибка = сумма расстояний."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    if k >= n:
        raise ValueError("k должно быть меньше числа объектов")

    D = _distances(X, X)

    medoids = [int(D.sum(axis=1).argmin())]
    for _ in range(1, k):
        cur_cost = D[:, medoids].min(axis=1)
        best_c, best_total = -1, np.inf
        for c in range(n):
            if c in medoids:
                continue
            total = np.minimum(cur_cost, D[:, c]).sum()
            if total < best_total:
                best_total, best_c = total, c
        medoids.append(best_c)

    cost = float(D[:, medoids].min(axis=1).sum())
    n_iter = 0

    for n_iter in range(1, max_iter + 1):
        labels = D[:, medoids].argmin(axis=1)
        new_medoids = list(medoids)
        for j in range(k):
            members = np.where(labels == j)[0]
            if len(members) == 0:
                continue
            sub = D[np.ix_(members, members)]
            new_medoids[j] = int(members[sub.sum(axis=1).argmin()])
        if new_medoids == medoids:
            break
        medoids = new_medoids

    for _ in range(refine):
        improved = False
        for j in range(k):
            best_swap, best_cost = None, cost
            others = [m for t, m in enumerate(medoids) if t != j]
            for c in range(n):
                if c in medoids:
                    continue
                cand = others + [c]
                new_cost = float(D[:, cand].min(axis=1).sum())
                if new_cost < best_cost - 1e-9:
                    best_cost, best_swap = new_cost, c
            if best_swap is not None:
                medoids[j] = best_swap
                cost = best_cost
                improved = True
        if not improved:
            break

    labels = D[:, medoids].argmin(axis=1)
    return {"labels": labels, "medoid_indices": np.array(medoids),
            "medoids": X[medoids],
            "error": float(D[:, medoids].min(axis=1).sum()), "n_iter": n_iter}
