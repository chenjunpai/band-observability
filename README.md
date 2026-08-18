# A band-observability criterion for sensor placement in continuous data assimilation of a chaotic two-dimensional flow

## 中文说明（全文）

本仓库是论文《A band-observability criterion for sensor placement in continuous data assimilation of a chaotic two-dimensional flow》的参考代码、冻结结果与复现环境。

### 这篇论文在做什么

连续资料同化中的 nudging 方法，已经知道：当观测“足够多”时，模型会指数地同步到真值。但实际问题不只是“够不够多”，而是“放在哪里”。这篇论文说明：真正决定网络好坏的量是**决定波段的可观测性 δ_K**，而不是传感器数量，也不是覆盖半径 h。

对于周期性条纹采样，这个机制还是**可以证明的**：如果只有 R 条等距的传感器行，那么在决定波段上，观测算子会留下一个维数为

`(2K+1)(2K+1−R)`

的零空间；这个零空间被 Shepard 插值算子精确地闭合。因此传感器间距必须满足 Nyquist 式的阈值。

### 目录结构

```text
├── nolab/           # 共享代码库
├── script/          # 冻结的实验脚本
├── environment.yml  # conda 环境
├── requirements.txt # pip 依赖
└── README.md
```

### 安装

```bash
conda env create -f environment.yml
conda activate neuralobserver
```

只有 learned-gain 那部分需要 torch；其余脚本只需要 numpy / scipy / matplotlib。

### 运行

```bash
python script/45_band_gram_spectrum.py   # 命题：核维数 33/22/11/0（秒级）
python script/48_tail_rebound_audit.py   # 尾半段反弹审计（读取已有曲线，约 2 秒）
python script/41_stripe_nyquist_v3.py    # 固定传感器数量的条纹 Nyquist 阶梯（约 100 分钟）
```

真值场缓存在仓库旁的 `truths/` 目录中；缺失时 `nolab.get_truth` 会自动重新生成。结果文件写入 `results/<name>/results.json`，并带有求解器指纹和来源信息。

### 主要结果

- `δ_K` 对误差平台的排序能力：Kendall `τ = 0.71`，AUC `= 0.98`；覆盖半径只达到 `0.52` / `0.88`。
- 没有任何 `δ_K ≤ 0` 的布局能同步（0 / 40）。
- 核维数 `(2K+1)(2K+1−R)` 在 `R = 8 / 9 / 10 / 11` 下验证为 `33 / 22 / 11 / 0`，其中 `K_c = 5`。
- 固定 `n = 784`、`T = 8` 的 22 行阶梯中，没有任何违反必要性的反例；阈值随黏性从 `9 → 11 → 15`。
- nudging 与 LETKF 的速率对比：`1.21` vs `2.60`，而 nudging 只有其 1/32 的求解器成本。

### 许可证

MIT。

---

# A band-observability criterion for sensor placement in continuous data assimilation of a chaotic two-dimensional flow

Reference code for the paper. Continuous data assimilation by nudging is known to converge when *enough*
observations are supplied, but network design is a question of *placement*; we show the controlling quantity
is the observability of the **determining band** `δ_K` — not the sensor count and not the coverage radius — and
that for periodic sampling the threshold is provable: `R` equidistant sensor rows leave a kernel of dimension
`(2K+1)(2K+1−R)` on the band, closed exactly by the Shepard interpolant.

## Layout

```arduino
├── nolab/           # shared library
├── script/          # frozen experiment scripts
├── environment.yml  # conda environment
├── requirements.txt # pip equivalent
└── README.md
```

## Install

```bash
conda env create -f environment.yml
conda activate neuralobserver
```

Only `script/35_learned_gain.py` needs torch; everything else uses numpy/scipy/matplotlib.

## Run

```bash
python script/45_band_gram_spectrum.py   # Proposition: kernel dimension 33/22/11/0 (seconds)
python script/48_tail_rebound_audit.py   # tail-half rebound audit (reads stored curves, ~2 s)
python script/41_stripe_nyquist_v3.py    # fixed-count stripe Nyquist (~100 min)
```

Truth fields are cached in a `truths/` directory next to the repository and regenerated automatically by
`nolab.get_truth` when missing. Result files are written to `results/<name>/results.json` with solver
provenance.

## Headline numbers

- `δ_K` ranks the plateau with Kendall `τ = 0.71` / `AUC = 0.98`; coverage radius gives `0.52` / `0.88`.
- No layout with `δ_K ≤ 0` synchronises (0 / 40).
- Kernel dimension `(2K+1)(2K+1−R)` verified as `33 / 22 / 11 / 0` at `R = 8 / 9 / 10 / 11`, `K_c = 5`.
- No counterexample among the 22-row fixed-`n = 784` ladder at `T = 8`, threshold moving `9 → 11 → 15`.
- nudging vs LETKF rate `1.21` vs `2.60` at 1/32 solver cost.

## License

MIT.
