# -*- coding: utf-8 -*-
"""
step4_report.py
===============
ПУНКТ 4 задания — формирование отчёта в формате PDF.

Обязательные разделы (по заданию):
    1. формулировка задания;
    2. гиперссылка на каталог репозитория с исходными текстами, наборами данных
       и др. сопутствующими материалами;
    3. рисунки с результатами визуализации;
    4. пояснения, раскрывающие смысл полученных результатов.

Запуск (после step1, step2, step3):
    python step4_report.py
"""

import os

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
REPO_URL = "https://github.com/wutong185/hierarchical_clustering"
OUT_NAME = "Отчет_Иерархическая_кластеризация.pdf"

pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))

styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Arial-Bold",
                    fontSize=16, textColor=colors.HexColor("#1a365d"), spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Arial-Bold",
                    fontSize=13, textColor=colors.HexColor("#2b6cb0"),
                    spaceBefore=12, spaceAfter=6)
H3 = ParagraphStyle("H3", parent=styles["Heading3"], fontName="Arial-Bold",
                    fontSize=11.5, textColor=colors.HexColor("#2b6cb0"),
                    spaceBefore=8, spaceAfter=4)
body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Arial",
                      fontSize=10.5, leading=15, alignment=TA_JUSTIFY, spaceAfter=6)
cap = ParagraphStyle("cap", parent=styles["BodyText"], fontName="Arial",
                     fontSize=9.5, leading=13, alignment=TA_CENTER,
                     spaceAfter=10, textColor=colors.HexColor("#555555"))
lst = ParagraphStyle("lst", parent=styles["BodyText"], fontName="Arial",
                     fontSize=9.5, leading=14, leftIndent=10, spaceAfter=3)


def P(t):
    return Paragraph(t, body)


def L(t):
    return Paragraph(t, lst)


def make_table(data, widths, font_size=9):
    tbl = Table(data, colWidths=widths)
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Arial"),
        ("FONTNAME", (0, 0), (-1, 0), "Arial-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#ebf4ff")),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]))
    return tbl


def add_image(story, path, width_mm, height_mm, caption):
    if os.path.exists(path):
        story.append(Paragraph(caption, cap))
        story.append(Image(path, width=width_mm * mm, height=height_mm * mm))
        story.append(Spacer(1, 8))


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Arial", 9)
    canvas.drawCentredString(A4[0] / 2, 12 * mm, f"— {doc.page} —")
    canvas.restoreState()


def main():
    comp = pd.read_csv(os.path.join(RES, "compare_with_partitional.csv"))
    summ = pd.read_csv(os.path.join(RES, "linkage_summary.csv"))
    hier = pd.read_csv(os.path.join(RES, "hier_error_vs_k.csv"))

    def s(metric, method):
        return float(summ.loc[summ.method == method, metric].iloc[0])

    ward_minus_km = comp["ward"] - comp["kmeans_sse"]
    avg_gap = float(ward_minus_km.mean())
    rel_gap = float((ward_minus_km / comp["kmeans_sse"] * 100).mean())

    story = []

    # ================= титул =================
    story.append(Paragraph("Отчет по заданию «Иерархическая кластеризация»", H1))
    story.append(Paragraph(
        "Набор данных: customers (сведения о клиентах банка, 850 объектов) — "
        "тот же, что в задании «Разделительная кластеризация»", H2))

    story.append(P(
        "Отчёт охватывает все четыре пункта задания: программу с дендрограммой "
        "(пункт 1), сравнение с результатами задания «Разделительная кластеризация» "
        "(пункт 2), сравнение мер сходства между собой (пункт 3) и настоящий отчёт "
        "(пункт 4)."))

    # ================= 1 =================
    story.append(Paragraph("1. Формулировка задания", H2))
    for i, txt in [
        (1, "Разработать программу, выполняющую иерархическую кластеризацию заданного "
            "набора данных. Параметры программы: набор данных и используемая мера "
            "сходства кластеров (<b>Single linkage, Complete linkage, Group average</b> "
            "или <b>Ward distance</b>). Мера сходства объектов произвольная (например, "
            "евклидово расстояние). Программа должна выполнять визуализацию результатов "
            "в виде <b>дендрограммы</b>."),
        (2, "Провести эксперименты на наборе данных из задания «Разделительная "
            "кластеризация». Сравнить полученные результаты с результатами того задания."),
        (3, "Провести эксперименты на выбранном наборе данных, варьируя меру сходства "
            "кластеров. Сравнить результаты, полученные для различных мер сходства, "
            "между собой."),
        (4, "Подготовить отчёт в формате PDF с разделами: формулировка задания; "
            "гиперссылка на каталог репозитория; рисунки с результатами визуализации; "
            "пояснения, раскрывающие смысл полученных результатов."),
    ]:
        story.append(L(f"<b>{i}.</b> {txt}"))

    # ================= 2 =================
    story.append(Paragraph("2. Гиперссылка на каталог репозитория", H2))
    story.append(P(
        "Исходные тексты программы, набор данных, таблицы результатов и рисунки "
        "размещены в репозитории GitHub: "
        '<link href="%s" color="#1a4fd6"><u>%s</u></link>' % (REPO_URL, REPO_URL)))
    story.append(P("Клонирование: <b>git clone %s.git</b>" % REPO_URL))
    story.append(P("Состав репозитория и соответствие пунктам задания:"))
    story.append(L("&nbsp;&nbsp;• <b>clustering_hier.py</b> — ядро: агломеративная "
                   "кластеризация по формуле Ланса–Уильямса для четырёх мер сходства, "
                   "обрезка дерева, SSE, копенетическая корреляция (пункты 1 и 3);"))
    story.append(L("&nbsp;&nbsp;• <b>dendro.py</b> — построение дендрограммы «с нуля» "
                   "(полной и усечённой, с раскраской по кластерам);"))
    story.append(L("&nbsp;&nbsp;• <b>step1_program.py</b> — пункт 1: демонстрация "
                   "программы и дендрограммы;"))
    story.append(L("&nbsp;&nbsp;• <b>step2_compare_partitional.py</b> — пункт 2: "
                   "сравнение с заданием «Разделительная кластеризация»;"))
    story.append(L("&nbsp;&nbsp;• <b>step3_linkage_compare.py</b> — пункт 3: "
                   "сравнение мер сходства;"))
    story.append(L("&nbsp;&nbsp;• <b>step4_report.py</b> — пункт 4: настоящий отчёт;"))
    story.append(L("&nbsp;&nbsp;• <b>partitional_reference.py</b> — k-Means и k-Medoids "
                   "из предыдущего задания (для корректного сравнения);"))
    story.append(L("&nbsp;&nbsp;• <b>data/</b> — customers.csv, custmers.txt, "
                   "partitional_baseline.csv (результаты предыдущего задания);"))
    story.append(L("&nbsp;&nbsp;• <b>results/</b> — таблицы (CSV) и рисунки (PNG)."))

    # ================= 3 =================
    story.append(PageBreak())
    story.append(Paragraph("3. Рисунки с результатами визуализации", H2))

    # --- 3.1 пункт 1 ---
    story.append(Paragraph("3.1. Пункт 1 — программа и дендрограмма", H3))
    story.append(P(
        "Программа запускается с параметрами «набор данных» и «мера сходства кластеров» "
        "(<b>python clustering_hier.py --method ward --k 5</b>). Расстояние между "
        "объектами — евклидово; данные предварительно стандартизованы. Алгоритм "
        "выполняет n−1 = 849 слияний, на каждом шаге объединяя пару кластеров с "
        "минимальным расстоянием и пересчитывая расстояния по формуле Ланса–Уильямса. "
        "Результат визуализируется в виде дендрограммы; обрезка дерева на k кластеров "
        "даёт готовое разбиение и значение ошибки."))

    add_image(story, os.path.join(RES, "fig1_dendrogram_truncated.png"), 150, 87,
              "Рис. 1. Дендрограмма (Ward distance), показаны последние 20 слияний; "
              "в скобках — число объектов в свёрнутом поддереве, цвет — кластеры при k = 5.")
    add_image(story, os.path.join(RES, "fig1_dendrogram_full.png"), 170, 42,
              "Рис. 2. Полная дендрограмма (все 850 листьев) — общий вид иерархической "
              "структуры набора customers.")

    story.append(P("Результат работы программы при обрезке дерева на k = 5:"))
    t1 = [["Мера сходства", "Высота корня", "SSE", "Размеры кластеров"]]
    for _, r in summ.iterrows():
        t1.append([r["method"].capitalize(), f"{r['root_height']:.2f}",
                   f"{r['sse_k5']:.2f}", r["sizes_k5"]])
    story.append(make_table(t1, [32 * mm, 28 * mm, 24 * mm, 66 * mm], font_size=8.5))
    story.append(Paragraph("Таблица 1. Что выдаёт программа для разных мер сходства "
                           "при k = 5.", cap))

    # --- 3.2 пункт 2 ---
    story.append(PageBreak())
    story.append(Paragraph("3.2. Пункт 2 — сравнение с заданием «Разделительная "
                           "кластеризация»", H3))
    story.append(P(
        "В предыдущем задании на том же наборе данных применялись k-Means и k-Medoids. "
        "Чтобы сравнение было корректным, перенесённый код k-Means/k-Medoids был запущен "
        "заново и сверен с сохранёнными значениями: максимальное расхождение составило "
        "<b>0.000031</b> для k-Means и <b>0.000041</b> для k-Medoids (различие только за "
        "счёт округления в CSV), то есть воспроизведение точное. Сравнение проводится по "
        "единой метрике — <b>SSE</b> (сумма квадратов расстояний до центроидов), которую "
        "минимизирует k-Means; для иерархической кластеризации SSE считается после "
        "обрезки дерева на k кластеров."))

    t2 = [["k", "k-Means", "Ward", "Complete", "Single", "Average"]]
    for _, r in comp.iterrows():
        t2.append([str(int(r["k"])), f"{r['kmeans_sse']:.1f}", f"{r['ward']:.1f}",
                   f"{r['complete']:.1f}", f"{r['single']:.1f}", f"{r['average']:.1f}"])
    story.append(make_table(t2, [14 * mm, 25 * mm, 25 * mm, 27 * mm, 25 * mm, 27 * mm],
                            font_size=8.5))
    story.append(Paragraph("Таблица 2. SSE при k = 3…9: k-Means и четыре меры сходства "
                           "иерархической кластеризации.", cap))

    add_image(story, os.path.join(RES, "fig2_sse_vs_k.png"), 148, 92,
              "Рис. 3. Зависимость SSE от числа кластеров: иерархическая кластеризация "
              "(4 меры сходства) и k-Means.")
    add_image(story, os.path.join(RES, "fig2_scatter_ward_vs_kmeans.png"), 160, 69,
              "Рис. 4. Визуальное сравнение разбиений при k = 5: Ward (слева) и "
              "k-Means (справа), проекция на две главные компоненты.")

    # --- 3.3 пункт 3 ---
    story.append(PageBreak())
    story.append(Paragraph("3.3. Пункт 3 — сравнение мер сходства между собой", H3))
    story.append(P(
        "Сравнивались четыре меры сходства на одном и том же наборе данных при "
        "одинаковой мере расстояния между объектами (евклидово). Помимо SSE "
        "использованы два показателя: <b>доля самого крупного кластера</b> при k = 5 "
        "(характеризует «эффект цепочки») и <b>копенетическая корреляция</b> — "
        "корреляция между расстояниями на дендрограмме и исходными расстояниями "
        "между объектами."))

    add_image(story, os.path.join(RES, "fig3_dendrograms.png"), 162, 104,
              "Рис. 5. Дендрограммы для четырёх мер сходства (последние 18 слияний, "
              "цвет — кластеры при k = 5).")

    story.append(PageBreak())
    t3 = [["Мера сходства", "Высота корня", "SSE (k=5)", "Доля крупнейшего кластера",
           "Копенетическая корреляция"]]
    for _, r in summ.iterrows():
        t3.append([r["method"].capitalize(), f"{r['root_height']:.2f}",
                   f"{r['sse_k5']:.2f}", f"{r['largest_share'] * 100:.1f}%",
                   f"{r['cophenetic']:.3f}"])
    story.append(make_table(t3, [30 * mm, 26 * mm, 24 * mm, 40 * mm, 40 * mm],
                            font_size=8.5))
    story.append(Paragraph("Таблица 3. Сравнение мер сходства (k = 5).", cap))

    add_image(story, os.path.join(RES, "fig3_sse_vs_k.png"), 145, 90,
              "Рис. 6. SSE в зависимости от k для четырёх мер сходства.")
    add_image(story, os.path.join(RES, "fig3_chaining.png"), 148, 78,
              "Рис. 7. Доля самого крупного кластера при k = 5 — количественная мера "
              "«эффекта цепочки».")
    add_image(story, os.path.join(RES, "fig3_cophenetic.png"), 148, 78,
              "Рис. 8. Копенетическая корреляция: насколько дендрограмма сохраняет "
              "исходные расстояния между объектами.")

    # ================= 4 =================
    story.append(PageBreak())
    story.append(Paragraph("4. Пояснения, раскрывающие смысл полученных результатов", H2))

    story.append(Paragraph("4.1. Условия эксперимента", H3))
    story.append(P(
        "Использован тот же набор customers, что и в задании «Разделительная "
        "кластеризация»: 850 объектов, 7 числовых признаков (возраст, образование, стаж, "
        "доход, задолженность по карте, прочие долги, отношение долга к доходу); признак "
        "Defaulted исключён из-за 150 пропусков. Данные стандартизованы (z-оценки), "
        "поэтому евклидово расстояние не зависит от масштаба признаков. "
        "Программа реализует агломеративную стратегию: начинаем с n кластеров-точек и на "
        "каждом шаге объединяем ближайшую пару, пока не останется один кластер."))

    story.append(Paragraph("4.2. Как устроен алгоритм", H3))
    story.append(P(
        "Ключевой момент — правило пересчёта расстояния от нового кластера до всех "
        "остальных (формула Ланса–Уильямса). При объединении кластеров s и t в u для "
        "любого другого кластера v расстояние вычисляется как линейная комбинация трёх "
        "известных величин d(s,v), d(t,v), d(s,t), и именно коэффициенты этой комбинации "
        "задают меру сходства:"))
    story.append(L("&nbsp;&nbsp;• <b>Single linkage</b>: d(u,v) = min(d(s,v), d(t,v)) — "
                   "расстояние между ближайшими точками кластеров;"))
    story.append(L("&nbsp;&nbsp;• <b>Complete linkage</b>: d(u,v) = max(d(s,v), d(t,v)) — "
                   "между самыми дальними точками;"))
    story.append(L("&nbsp;&nbsp;• <b>Group average</b>: средневзвешенное по размерам "
                   "кластеров расстояние между всеми парами точек;"))
    story.append(L("&nbsp;&nbsp;• <b>Ward distance</b>: объединяется пара, дающая "
                   "минимальный прирост внутрикластерной суммы квадратов, поэтому "
                   "расстояния пересчитываются через квадраты, а высота слияния — "
                   "корень из полученной величины."))
    story.append(P(
        "Для n = 850 объектов строится 849 слияний; вычислительная сложность — O(n³) "
        "операций с матрицей расстояний (на практике около 0.2 с на одну меру сходства). "
        "Дендрограмма строится «с нуля»: для каждого слияния рисуется П-образная связь "
        "между поддеревьями на высоте этого слияния, а порядок листьев задаётся обходом "
        "дерева."))

    story.append(Paragraph("4.3. Сравнение с разделительной кластеризацией (пункт 2)", H3))
    story.append(P(
        f"Иерархическая кластеризация при любой мере сходства даёт SSE хуже, чем k-Means: "
        f"у лучшей меры (Ward) превышение составляет в среднем "
        f"<b>{avg_gap:.1f}</b> ({rel_gap:.1f}%) на всём диапазоне k = 3…9. Это "
        f"закономерно и не означает «худший алгоритм»: k-Means <i>непосредственно "
        f"минимизирует</i> SSE, тогда как иерархическая кластеризация строит единое "
        f"дерево и при обрезке на k кластеров не может пересматривать ранее принятые "
        f"слияния — разбиение жёстко задано структурой дерева. Зато иерархический подход "
        f"имеет принципиальное преимущество: он не требует заранее задавать число "
        f"кластеров, а дендрограмма показывает всю структуру сразу, позволяя выбрать k "
        f"после построения дерева."))
    story.append(P(
        "Визуально (рис. 4) разбиения Ward и k-Means при k = 5 похожи: оба выделяют "
        "компактное ядро «обычных» клиентов, группу с высоким доходом и небольшую группу "
        "клиентов с большой задолженностью. Различие в том, что Ward дополнительно "
        "дробит крупное ядро по стажу и возрасту, а k-Means — по доходу, поскольку именно "
        "доход даёт наибольший вклад в SSE."))

    story.append(Paragraph("4.4. Сравнение мер сходства (пункт 3)", H3))
    story.append(P(
        f"<b>Single и Group average демонстрируют «эффект цепочки» (chaining).</b> "
        f"При k = 5 самый крупный кластер содержит "
        f"{s('largest_share', 'single') * 100:.1f}% и "
        f"{s('largest_share', 'average') * 100:.1f}% объектов соответственно "
        f"(рис. 7): фактически выделяется один огромный кластер и несколько отдельных "
        f"точек-выбросов. Причина в самом правиле: Single смотрит только на ближайшую "
        f"пару точек, поэтому легко «прицепляет» к кластеру отдельные точки, лежащие "
        f"вдоль некоторой цепочки, — расстояние между крайними точками такого кластера "
        f"может быть огромным. У Group average эффект слабее, но на этом наборе "
        f"сохраняется."))
    story.append(P(
        f"<b>Complete linkage</b> даёт заметно более сбалансированное разбиение "
        f"(крупнейший кластер {s('largest_share', 'complete') * 100:.1f}%) и лучшее, чем у "
        f"Single/Average, значение SSE, но склонен к излишнему дроблению: при k = 5 "
        f"получаются кластеры размером 578, 260, 5, 4 и 3 — три последних фактически "
        f"являются выбросами."))
    story.append(P(
        f"<b>Ward distance</b> показывает лучшие результаты по обоим «качественным» "
        f"показателям: минимальный SSE ({s('sse_k5', 'ward'):.1f} против "
        f"{s('sse_k5', 'single'):.1f} у Single) и наиболее сбалансированные размеры "
        f"кластеров (({summ.loc[summ.method == 'ward', 'sizes_k5'].iloc[0]})), то есть "
        f"доля крупнейшего кластера всего {s('largest_share', 'ward') * 100:.1f}%. "
        f"Объяснение: критерий Уорда минимизирует именно внутрикластерный разброс, "
        f"поэтому он «предпочитает» компактные сферические кластеры примерно равного "
        f"размера и не страдает от цепочек."))
    story.append(P(
        f"<b>Обратный результат по копенетической корреляции</b> (рис. 8) стоит "
        f"прокомментировать отдельно, так как он может показаться противоречием: "
        f"наивысшая корреляция у Single ({s('cophenetic', 'single'):.3f}) и Average "
        f"({s('cophenetic', 'average'):.3f}), а у Ward — наименьшая "
        f"({s('cophenetic', 'ward'):.3f}). Причина в том, что показатели измеряют разное. "
        f"Копенетическая корреляция говорит лишь о том, насколько <i>высоты слияний "
        f"пропорциональны исходным расстояниям</i>: Single и Average монотонно следуют "
        f"за расстояниями, поэтому воспроизводят их точно. Ward же оперирует приростом "
        f"внутрикластерной суммы квадратов, и его высоты — не расстояния, а мера "
        f"«разброса»; за это он платит низкой корреляцией с исходной метрикой. "
        f"Но качество самого разбиения (SSE, сбалансированность кластеров) от этого не "
        f"страдает — оно у Ward лучше. Иными словами, высокая копенетическая корреляция "
        f"является необходимым, но не достаточным условием хорошей кластеризации."))

    story.append(Paragraph("4.5. Выводы", H3))
    story.append(P(
        "1. Реализован агломеративный алгоритм иерархической кластеризации «с нуля»: "
        "четыре меры сходства (Single, Complete, Group average, Ward), евклидово "
        "расстояние между объектами, построение дендрограммы, обрезка дерева на "
        "произвольное k и расчёт ошибки — пункт 1 выполнен."))
    story.append(P(
        f"2. Проведено сравнение с заданием «Разделительная кластеризация» на том же "
        f"наборе customers: k-Means даёт SSE меньше, чем любая иерархическая мера "
        f"(у лучшей, Ward, — в среднем на {rel_gap:.1f}%), что объясняется различием "
        f"оптимизируемых критериев, а преимущество иерархического подхода — возможность "
        f"выбрать число кластеров после построения дерева — пункт 2 выполнен."))
    story.append(P(
        f"3. Сравнение мер сходства показало: Single и Group average страдают «эффектом "
        f"цепочки» (крупнейший кластер 99.5% и 98.0% объектов), Complete склонен к "
        f"дроблению, а Ward даёт самые компактные и сбалансированные кластеры при "
        f"наименьшем SSE — пункт 3 выполнен."))
    story.append(P(
        "4. Оформлен настоящий отчёт с гиперссылкой на репозиторий, рисунками и "
        "пояснениями — пункт 4 выполнен."))

    # ---------- сборка ----------
    out_path = os.path.join(BASE, OUT_NAME)
    if os.path.exists(out_path):
        try:
            with open(out_path, "ab"):
                pass
        except OSError:
            alt = OUT_NAME.replace(".pdf", "_обновленный.pdf")
            out_path = os.path.join(BASE, alt)
            print("ВНИМАНИЕ: исходный PDF занят другой программой.")
            print("          Сохраняем отчёт как:", alt)

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            rightMargin=20 * mm, leftMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm,
                            title="Отчет: Иерархическая кластеризация",
                            author="wutong185")
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print("Отчёт сформирован:", out_path)


if __name__ == "__main__":
    main()
