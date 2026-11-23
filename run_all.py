# -*- coding: utf-8 -*-
"""
run_all.py
==========
一键执行全部 4 个 пунктов задания «Иерархическая кластеризация».

    1) step1_program.py            — пункт 1: программа + дендрограмма
    2) step2_compare_partitional.py— пункт 2: сравнение с «Разделительной кластеризацией»
    3) step3_linkage_compare.py    — пункт 3: сравнение мер сходства
    4) step4_report.py             — пункт 4: PDF-отчёт

Запуск:
    python run_all.py
    python run_all.py --only 1 2   # только первые два пункта
"""

import argparse
import os
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    (1, "step1_program.py", "ПУНКТ 1 — программа и дендрограмма"),
    (2, "step2_compare_partitional.py",
     "ПУНКТ 2 — сравнение с заданием «Разделительная кластеризация»"),
    (3, "step3_linkage_compare.py", "ПУНКТ 3 — сравнение мер сходства"),
    (4, "step4_report.py", "ПУНКТ 4 — формирование PDF-отчёта"),
]


def main():
    ap = argparse.ArgumentParser(description="Прогнать все пункты задания")
    ap.add_argument("--only", nargs="*", type=int, default=None,
                    help="номера пунктов, например: --only 1 2")
    args = ap.parse_args()

    steps = STEPS
    if args.only:
        steps = [s for s in STEPS if s[0] in args.only]

    t_all = time.time()
    failed = []

    for num, script, title in steps:
        print("\n" + "#" * 78)
        print(f"# ПУНКТ {num}/{len(STEPS)}: {title}")
        print(f"# файл: {script}")
        print("#" * 78)

        t0 = time.time()
        r = subprocess.run([sys.executable, script], cwd=BASE)
        dt = time.time() - t0

        if r.returncode != 0:
            failed.append((script, r.returncode))
            print(f"!!! ОШИБКА в {script} (код {r.returncode})")
        else:
            print(f"--- готово за {dt:.1f} c")

    print("\n" + "=" * 78)
    print("ИТОГ")
    print("=" * 78)
    if failed:
        print("Завершились с ошибкой:")
        for s, c in failed:
            print(f"  {s} (код {c})")
        sys.exit(1)

    print(f"Все пункты выполнены успешно за {time.time() - t_all:.1f} c.\n")
    print("Результаты в папке results/:")
    res = os.path.join(BASE, "results")
    if os.path.isdir(res):
        for name in sorted(os.listdir(res)):
            p = os.path.join(res, name)
            print(f"  {name:<44} {os.path.getsize(p) / 1024:>8.1f} KB")
    print(f"\nОтчёт: {os.path.join(BASE, 'Отчет_Иерархическая_кластеризация.pdf')}")


if __name__ == "__main__":
    main()
