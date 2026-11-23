# 层次聚类作业（Иерархическая кластеризация）

数据挖掘课程第四份作业：**手写凝聚式（agglomerative）层次聚类**，
四种链接方式：**Single / Complete / Group average / Ward**，
输出**树状图**，并与上次的划分式聚类对比。

> 报告（PDF）和程序输出是俄语（交给老师）；本说明是中文，给你自己看。

---

## 一、4 个小编号 → 对应文件

| 题号 | 要求 | 对应脚本 | 产出 |
|---|---|---|---|
| **题 1** | 程序：层次聚类，参数=数据集+链接方式（4 种），欧氏距离，**树状图** | `clustering_hier.py`（核心）<br>`dendro.py`（画树）<br>`step1_program.py`（演示） | 图 1、图 2、`clusters_ward_k5.csv` |
| **题 2** | 在**上次那个数据集**（customers）上实验，并**和划分式聚类结果对比** | `step2_compare_partitional.py` | 图 3、图 4、`compare_with_partitional.csv` |
| **题 3** | 换**不同链接方式**做实验，互相比较 | `step3_linkage_compare.py` | 图 5–8、`linkage_summary.csv` |
| **题 4** | PDF 报告（4 章节） | `step4_report.py` | `Отчет_Иерархическая_кластеризация.pdf` |

每个脚本开头打印 `ПУНКТ N`，单跑就是那一题的成果。

---

## 二、数据集（复用上次的）

`data/customers.csv`：**850 行 × 10 列**（银行客户），作业第 2 条明确要求用这份。

- 聚类用 **7 个数值特征**：Age、Education、YearsEmployed、Income、CardDebt、OtherDebt、DebtIncomeRatio
- `Row`、`CustomerId` 是标识符，排除
- `Defaulted` 有 **150 个缺失（17.6%）**，排除
- 数据做 **z-score 标准化**

`data/partitional_baseline.csv` 是**上次作业**（k-Means / k-Medoids）的结果，用来做第 2 条的对比。我重新跑了一遍验证能复现，**最大偏差 0.00003**（只是 CSV 四位小数舍入）。

---

## 三、文件夹结构

```
1123数据挖掘/
├─ clustering_hier.py            ★ 核心：4 种链接方式 + 剪枝取 k + SSE + 共表型相关
├─ dendro.py                     手写树状图（完整版 + 截断版 + 按簇上色）
├─ partitional_reference.py      上次的 k-Means / k-Medoids（原样搬来对比用）
├─ viz.py                        散点图、折线图、柱状图
├─ step1_program.py              题 1：程序演示 + 树状图
├─ step2_compare_partitional.py  题 2：与划分式聚类对比
├─ step3_linkage_compare.py      题 3：4 种链接方式互比
├─ step4_report.py               题 4：生成 PDF 报告
├─ run_all.py                    一键跑完 4 题
├─ data/  customers.csv, custmers.txt, partitional_baseline.csv
├─ results/                      CSV 表格 + PNG 图
└─ Отчет_Иерархическая_кластеризация.pdf
```

---

## 四、怎么运行

```powershell
cd C:\Users\HP\Desktop\1123数据挖掘
python run_all.py          # 一键全跑，约 15 秒
```

单独跑：
```powershell
python step1_program.py             # 题 1
python step2_compare_partitional.py # 题 2
python step3_linkage_compare.py     # 题 3
python step4_report.py              # 题 4（报告）

# 直接调程序（验证"参数化"）
python clustering_hier.py --method ward --k 5
python clustering_hier.py --method single
python clustering_hier.py --data data/customers.csv --method complete --k 7
```

参数：`--data`（数据集）、`--method`（single/complete/average/ward）、`--k`（剪枝后的簇数）。

依赖：`numpy pandas matplotlib reportlab`（本机 Python 3.12 已装好）

---

## 五、结果

### 1）四种链接方式对比（k=5）

| 链接方式 | 树根高度 | SSE | 最大簇占比 | 共表型相关 | 簇大小 |
|---|---|---|---|---|---|
| Single | 6.74 | 5433.41 | **99.5%** | 0.836 | [846,1,1,1,1] |
| Complete | 14.78 | 3992.25 | 68.0% | 0.660 | [578,260,5,4,3] |
| Average | 10.49 | 4781.81 | **98.0%** | 0.852 | [833,9,4,3,1] |
| **Ward** | **53.73** | **3096.01** | **58.2%** | 0.496 | [495,157,105,12,81] |

### 2）和上次 k-Means 比（SSE，越低越好）

| k | k-Means | Ward | Complete | Single | Average |
|---|---|---|---|---|---|
| 3 | 3702.6 | 3900.1 | 4980.4 | 5624.4 | 5191.0 |
| 4 | 3249.8 | 3463.2 | 4038.4 | 5442.3 | 4828.0 |
| 5 | 2881.4 | 3096.0 | 3992.3 | 5433.4 | 4781.8 |
| 6 | 2603.9 | 2826.2 | 3649.1 | 5178.0 | 4760.6 |
| 7 | 2430.4 | 2601.6 | 3607.8 | 5170.8 | 4717.1 |
| 8 | 2274.7 | 2474.9 | 3445.6 | 5094.6 | 4156.5 |
| 9 | 2152.8 | 2364.2 | 3269.5 | 5046.4 | 4112.2 |

### 3）核心结论

1. **Ward 是四种里最好的**：SSE 最低、簇最均衡（最大簇只占 58%）。
2. **Single 和 Average 有"链式效应"**：k=5 时最大簇占了 **99.5% / 98.0%**，
   等于"一个巨型簇 + 几个孤立点"，基本不可用。原因是 Single 只看最近点对，
   容易沿着一条链把远处的点也拉进来。
3. **层次聚类 SSE 比 k-Means 略差**（Ward 平均高 **7.7%**）——这不奇怪：
   k-Means 是**直接优化 SSE** 的，而层次聚类一旦合并就不能撤销，
   剪枝得到的划分被树结构限制住了。
4. **但层次聚类有 k-Means 没有的优势**：不需要预先给 k，一根树状图看全部结构，
   k 可以事后选。这是两种方法的核心取舍。
5. **Ward 的共表型相关最低（0.496）不代表它差**，详见下面 FAQ。

---

## 六、结果展示看哪些图

| 文件 | 题号 | 展示什么 |
|---|---|---|
| `results/fig1_dendrogram_truncated.png` | 题 1 | **树状图**（最后 20 次合并，可读版），括号里是子树大小 |
| `results/fig1_dendrogram_full.png` | 题 1 | 完整树状图（850 个叶子） |
| `results/fig2_sse_vs_k.png` | 题 2 | SSE vs k：4 种分层 + k-Means |
| `results/fig2_sse_bars.png` | 题 2 | 各 k 下 SSE 柱状对比 |
| `results/fig2_scatter_ward_vs_kmeans.png` | 题 2 | Ward vs k-Means 散点对比（PCA 2D） |
| `results/fig3_dendrograms.png` | 题 3 | **4 种链接方式的树状图并排** |
| `results/fig3_sse_vs_k.png` | 题 3 | 4 种链接的 SSE vs k |
| `results/fig3_chaining.png` | 题 3 | 最大簇占比（链式效应量化） |
| `results/fig3_cophenetic.png` | 题 3 | 共表型相关系数 |
| `Отчет_Иерархическая_кластеризация.pdf` | 题 4 | 10 页报告，4 章节齐全 |

> 树状图为什么有"截断版"？850 个叶子画出来就是一团黑，所以截断版只画最后
> 18–20 次合并，把下面的子树折叠成一个叶子标上 `(大小)`，这样才看得清。

---

## 七、算法实现要点（答辩可能问）

**核心是 Lance–Williams 公式**：合并 s、t 成 u 后，u 到任意 v 的距离是
d(s,v)、d(t,v)、d(s,t) 的线性组合，不同的系数就是不同的链接方式：

| 链接方式 | 更新规则 |
|---|---|
| Single | `d(u,v) = min(d(s,v), d(t,v))` |
| Complete | `d(u,v) = max(d(s,v), d(t,v))` |
| Average | `d(u,v) = (nₛ·d(s,v) + nₜ·d(t,v)) / (nₛ+nₜ)` |
| Ward | `d²(u,v) = [(nᵥ+nₛ)d²(s,v) + (nᵥ+nₜ)d²(t,v) − nᵥ·d²(s,t)] / (nᵥ+nₛ+nₜ)` |

实现细节：
- 用 850×850 矩阵维护距离，合并后把那一行/列设成 `inf`，每次 `argmin` 找最近对
- Ward 在**平方距离**上运算，树高取 `sqrt`
- 850 个点 → 849 次合并，一次约 **0.2 秒**
- 剪枝：union-find，执行前 `n−k` 次合并，剩下的根就是 k 个簇
- 树状图**自己画**（没用 scipy）：递归给叶子分配 x 坐标，每次合并画一个 "П" 形

---

## 八、常见问题

**Q：为什么 Ward 的共表型相关只有 0.496，却是最好的？**
A：两个指标测的不是一回事。共表型相关衡量"树的合并高度和原始距离有多成比例"。
Single/Average 的合并高度**就是距离**，所以能做到 0.84–0.85；
Ward 的合并高度是**簇内方差增量**，本来就不是距离，所以相关性天然低。
但 Ward 的划分质量（SSE、均衡度）反而最好。**高共表型相关是必要条件，不是充分条件**——
Single 有 0.836 的相关性，却给出了 99.5% 一个簇的废结果。

**Q：为什么不用 scipy.cluster.hierarchy？**
A：作业写的是"разработайте программу"（自己开发），而且手写能讲清 Lance–Williams
公式；树状图也自己画，答辩时更好解释。

**Q：SSE 比 k-Means 差，是不是说明层次聚类不行？**
A：不是。k-Means 的优化目标**就是** SSE，层次聚类优化的是树结构。
层次聚类换来的是：不用预设 k、能看到完整的层次结构、结果确定（不像 k-Means 受初始化影响）。

**Q：跑一次多久？**
A：全部约 15 秒。单次层次聚类 0.2 秒左右（4 种链接都算完不到 1 秒）。
