# -*- coding: utf-8 -*-
"""
dendro.py
=========
Построение ДЕНДРОГРАММЫ «с нуля» (без scipy.cluster.hierarchy.dendrogram).

Дендрограмма — обязательный элемент пункта 1 задания: результат иерархической
кластеризации должен быть визуализирован в виде дендрограммы.

Поддерживается:
    * полная дендрограмма (все n листьев);
    * усечённая (truncate = m) — показываются только последние m слияний,
      а нижние поддеревья сворачиваются в один лист с подписью размера кластера;
    * раскраска ветвей по k кластерам (обрезка дерева на k кластеров).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from clustering_hier import BASE_DIR, METHOD_SHORT

plt.rcParams["font.family"] = "Arial"
plt.rcParams["axes.unicode_minus"] = False

PALETTE = ["#e74c3c", "#2980b9", "#27ae60", "#f39c12", "#8e44ad",
           "#16a085", "#d35400", "#2c3e50", "#c0392b"]


def _build_tree(result):
    """children[node] = (c1, c2), height[node] = высота слияния, size[node] = размер."""
    n = result["n"]
    children, height, size = {}, {}, {}
    for m, (c1, c2, h, sz) in enumerate(result["merges"]):
        node = n + m
        children[node] = (c1, c2)
        height[node] = h
        size[node] = sz
    return children, height, size


def _cluster_of_leaves(n, labels_by_leaf):
    return labels_by_leaf


def draw_dendrogram(result, labels, title, filename,
                    truncate=None, figsize=(7.4, 4.6), dpi=150,
                    show_heights=True):
    """
    Рисует дендрограмму.

    :param result:  результат agglomerative(...)
    :param labels:  метки кластеров после обрезки дерева (для раскраски ветвей)
    :param truncate: None — полное дерево; m — показать только m последних слияний
    :param figsize: размер рисунка в дюймах
    """
    n = result["n"]
    children, height, size = _build_tree(result)
    root = 2 * n - 2

    do_not_show = (n - 1 - truncate) if truncate else None

    # ---------- разметка листьев по горизонтали ----------
    xpos, order = {}, []

    def walk(node):
        if node < n:
            x = len(order)
            order.append(node)
            xpos[node] = float(x)
            return float(x)
        if do_not_show is not None and (node - n) < do_not_show:
            # поддерево сворачивается в один лист
            x = len(order)
            order.append(node)
            xpos[node] = float(x)
            return float(x)
        c1, c2 = children[node]
        x1 = walk(c1)
        x2 = walk(c2)
        x = (x1 + x2) / 2.0
        xpos[node] = x
        return x

    walk(root)

    def height_of(node):
        return height.get(node, 0.0)

    def color_of(node):
        """Цвет ветви = кластер самого левого листа поддерева."""
        while node >= n:
            node = children[node][0]
        return PALETTE[labels[node] % len(PALETTE)]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ---------- рисуем «U»-образные связи ----------
    for node in list(children.keys()):
        if do_not_show is not None and (node - n) < do_not_show:
            continue
        c1, c2 = children[node]
        y = height_of(node)
        x1, x2 = xpos[c1], xpos[c2]
        col = color_of(node)
        lw = 1.6 if truncate else 0.6
        ax.plot([x1, x1], [height_of(c1), y], color=col, lw=lw, solid_capstyle="butt")
        ax.plot([x2, x2], [height_of(c2), y], color=col, lw=lw, solid_capstyle="butt")
        ax.plot([x1, x2], [y, y], color=col, lw=lw, solid_capstyle="butt")

    # ---------- подписи листьев ----------
    if truncate:
        for node in order:
            if node >= n:
                ax.text(xpos[node], -0.01 * height[root], f"({size.get(node, n)})",
                        ha="center", va="top", fontsize=8, rotation=0)
            else:
                ax.text(xpos[node], -0.01 * height[root], f"{node}",
                        ha="center", va="top", fontsize=6, rotation=90)
        ax.set_xlabel("Кластеры (в скобках — число объектов в свёрнутом поддереве)")
    else:
        ax.set_xlabel("Объекты (в порядке обхода дерева)")

    ax.set_ylabel("Высота слияния (мера сходства)")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.22, axis="y")
    if show_heights:
        ax.margins(x=0.01)
    fig.tight_layout()

    path = os.path.join(BASE_DIR, "results", filename)
    fig.savefig(path)
    plt.close(fig)
    return path


def draw_dendrogram_panels(results, labels_list, title, filename,
                           truncate=18, figsize=(12.4, 8.0), dpi=150):
    """
    Сетка 2×2: дендрограммы для нескольких мер сходства (пункт 3 задания).
    """
    nres = len(results)
    ncols = 2
    nrows = int(np.ceil(nres / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    axes = np.atleast_1d(axes).ravel()

    for ax, result, labels in zip(axes, results, labels_list):
        n = result["n"]
        children, height, size = _build_tree(result)
        root = 2 * n - 2
        do_not_show = n - 1 - truncate

        xpos, order = {}, []

        def walk(node, children=children):
            if node < n or (node - n) < do_not_show:
                x = len(order)
                order.append(node)
                xpos[node] = float(x)
                return float(x)
            c1, c2 = children[node]
            x1 = walk(c1)
            x2 = walk(c2)
            x = (x1 + x2) / 2.0
            xpos[node] = x
            return x

        walk(root)

        def height_of(node):
            return height.get(node, 0.0)

        def color_of(node):
            while node >= n:
                node = children[node][0]
            return PALETTE[labels[node] % len(PALETTE)]

        for node in children:
            if (node - n) < do_not_show:
                continue
            c1, c2 = children[node]
            y = height_of(node)
            x1, x2 = xpos[c1], xpos[c2]
            col = color_of(node)
            ax.plot([x1, x1], [height_of(c1), y], color=col, lw=1.5)
            ax.plot([x2, x2], [height_of(c2), y], color=col, lw=1.5)
            ax.plot([x1, x2], [y, y], color=col, lw=1.5)

        for node in order:
            sz = size.get(node, 1) if node >= n else 1
            ax.text(xpos[node], -0.02 * height[root], f"({sz})" if node >= n else " ",
                    ha="center", va="top", fontsize=6.5)

        ax.set_title(f"{METHOD_SHORT[result['method']]}: корень на высоте "
                     f"{result['heights'][-1]:.2f}", fontsize=10)
        ax.set_ylabel("Высота слияния")
        ax.grid(alpha=0.2, axis="y")
        ax.set_xticks([])

    for ax in axes[nres:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(BASE_DIR, "results", filename)
    fig.savefig(path)
    plt.close(fig)
    return path
