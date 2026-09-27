# CondOT Flow Matching 与 VP-SDE ODE 求解器实验

本项目从零实现并比较两类生成概率路径上的 ODE 求解器：**CondOT Flow Matching**（直线概率路径）与 **VP-SDE probability-flow ODE**（方差保持扩散的确定性概率流路径）。

目标不是调用第三方采样器，而是在相同的小型 DiT 模型、相同初始噪声、相同条件和相同积分预算下，比较网络的不同预测目标及各种 ODE 求解器的数值误差、NFE、耗时和生成分布质量。

> 这里使用 encoder 一次、DiT decoder 多次的二维四峰 toy 模型做可控实验。绝对 W2 数值不能直接外推到真实自动驾驶任务；但参数化的条件数、误差传播和求解器匹配关系具有工程参考意义。

## 1. 目录与模块分工

```text
DPMSolver++2m/
├── schedules/probability_paths.py  # CondOT / VP-SDE 路径、时间网格、所有参数化转换
├── solvers/                        # 每个 ODE 方法一个类/文件
│   ├── euler.py, midpoint.py, heun.py, rk4.py, ab2.py
│   ├── condot_2m.py                # CondOT 结构化二阶多步法
│   ├── flow_matching_solvers.py    # CondOT 统一入口
│   ├── vp_explicit.py, vp_dpm_solver.py, vp_sde_solvers.py
├── model/                          # toy data、encoder/DiT decoder、质量指标
├── reports/tables.py               # CSV、Markdown、表格图片工具
├── docs/                           # 中文数学推导与可运行示例
├── train/train.py, train/train_vp.py # 两条路径的单独训练
├── train_all.py                    # 一键训练所有输出参数化
├── benchmark.py / benchmark_vp.py  # 单 checkpoint 基准测试
└── run_all.py                      # 一键导出统计、图表和样本可视化
```

每条采样轨迹只运行一次 encoder 来缓存条件上下文。每个 NFE 只额外运行一次 decoder，因此 `NFE` 可直接理解为单个样本的 decoder 调用次数。

## 2. 统一数学问题

所有方法都求解确定性常微分方程：

$$
\frac{d x_t}{dt}=u_\theta(x_t,t,c),
$$

其中 $x_t$ 是当前样本状态，$c$ 是条件，$u_\theta$ 是由网络输出转换得到的 ODE 速度场。概率路径、直接预测的量及积分公式不同，但最终均落到这个问题。

### 2.1 CondOT / 直线 Flow Matching

训练路径为

$$
x_t=(1-t)x_{\mathrm{noise}}+t x_{\mathrm{data}},\qquad t\in[0,1],
$$

因此条件速度目标为

$$
u_t=x_{\mathrm{data}}-x_{\mathrm{noise}}.
$$

CondOT 模型可直接预测 `velocity`，也可预测 clean data $D_t$ 或条件 score $s_t=\nabla_x\log p_t(x_t\mid c)$。三者的严格关系为

$$
u_t=\frac{D_t-x_t}{1-t},\qquad D_t=x_t+(1-t)u_t,
$$

$$
D_t=\frac{x_t+(1-t)^2s_t}{t},
\qquad
u_t=\frac{x_t+(1-t)s_t}{t}.
$$

data 转 velocity 在 $t\to1$ 病态，score 转 velocity 在 $t\to0$ 病态。为与训练保持一致，默认在 decoder 输入端将时间裁剪至 $[0.01,0.99]$；物理积分路径仍覆盖完整区间。

### 2.2 VP-SDE probability-flow ODE

VP 路径写成

$$
x_t=\alpha_t x_0+\sigma_t\epsilon,
\qquad \alpha_t^2+\sigma_t^2=1.
$$

其确定性 probability-flow ODE 为

$$
\frac{d x_t}{dt}
=-\frac{1}{2}\beta(t)x_t-\frac{1}{2}\beta(t)s_\theta(x_t,t).
$$

若网络预测噪声，则 $s_\theta=-\epsilon_\theta/\sigma_t$，所以

$$
u_t=-\frac{1}{2}\beta(t)x_t
+\frac{1}{2}\beta(t)\frac{\epsilon_\theta}{\sigma_t}.
$$

本项目支持 `noise`、`x_start`、`score` 与 `pf_velocity`。后者是上式的 **probability-flow ODE drift**，并非 DDPM 常用的 latent-$v$ 参数化。转换由 `schedules/probability_paths.py` 统一实现，例如

$$
\epsilon_\theta=\sigma_t\left(x_t+\frac{2u_\theta}{\beta(t)}\right),
\qquad
\hat{x}_0=\frac{x_t-\sigma_t\epsilon_\theta}{\alpha_t}.
$$

## 3. 已实现求解器

| 求解器全称 | 命令名 | 阶数 | 每区间新增 decoder 调用 | 说明 |
|---|---|---:|---:|---|
| 显式欧拉法 | `euler` | 1 | 1 | 最低成本基线 |
| 显式中点法 | `midpoint` | 2 | 2 | 二阶 Runge--Kutta |
| Heun 显式梯形法 | `heun` | 2 | 2 | predictor-corrector 二阶法 |
| 经典四阶 Runge--Kutta 法 | `rk4` | 4 | 4 | 高精度数值参考 |
| 二阶 Adams--Bashforth 法 | `ab2` | 2 | 1 | 首步欧拉，随后复用历史速度 |
| CondOT 二阶多步法 | `condot_2m` | 2 | 1 | 对预测 $D_t=x_t+(1-t)u_t$ 结构化外推 |
| DPM-Solver 二阶多步法 | `dpmsolver_2m` | 2 | 约 1 | VP 下主要适合 noise/score |
| DPM-Solver++ 二阶多步法 | `dpmsolverpp_2m` | 2 | 约 1 | VP 下主要适合 data/$x_0$ |

`midpoint`、`heun`、`rk4` 用更多 NFE 换精度；`ab2` 和二阶 DPM 方法在速度场平滑时通常有更好的固定预算性价比。

推导笔记在 `docs/`：`00_overview.ipynb` 总览；`01` 至 `05` 讲经典积分方法；`06_condot_2m.ipynb` 从半线性结构、积分因子、$\lambda$ 空间到二阶外推完整推导 CondOT 2M；`07_vpsde_dpm_solver.ipynb` 推导 VP-SDE probability-flow ODE、四种输出参数化、加权训练和 DPM-Solver / DPM-Solver++ 的匹配关系。

## 4. 训练中的参数化对齐

不同输出的原始 MSE 不同。项目默认采用**速度等价加权损失**：以输出扰动传到真实 ODE velocity 的一阶雅可比平方作为权重，从而更公平地优化最终积分误差。

CondOT 权重为

$$
w_{\mathrm{velocity}}=1,
\qquad w_{\mathrm{data}}=\frac{1}{(1-t)^2},
\qquad w_{\mathrm{score}}=\left(\frac{1-t}{t}\right)^2.
$$

VP-SDE 权重为

$$
w_{\epsilon}=\left(\frac{\beta}{2\sigma}\right)^2,
\quad w_{x_0}=\left(\frac{\beta\alpha}{2\sigma^2}\right)^2,
\quad w_{u}=1,
\quad w_s=\left(\frac{\beta}{2}\right)^2.
$$

为了避免理论端点的无穷权重，默认使用 `--training-endpoint-eps 1e-2`、`--weight-clip 100`，并在 batch 内把权重归一为均值 1。加权只对齐“速度误差”，并不能消除网络容量、目标平滑性和参数变换条件数造成的差异。

## 5. 如何训练与运行

```bash
cd /home/zhangjinrui/DockerDisk/code_files/代码/端到端/秋招同步/freetect/DPMSolver++2m
```

一键训练 CondOT 的 `velocity/data/score` 与 VP-SDE 的 `noise/x_start/pf_velocity/score`：

```bash
python train_all.py --device cuda --train-steps 3200 --batch-size 1024
```

也可单独训练：

```bash
python train/train.py --device cuda --train-steps 3200 --batch-size 1024 \
  --prediction-type all --checkpoint checkpoints/toy_condot.pt

python train/train_vp.py --device cuda --train-steps 3200 --batch-size 1024 \
  --checkpoint checkpoints/toy_vpsde.pt
```

单 checkpoint 的数值基准只加载模型、不训练：

```bash
python benchmark/benchmark.py --device cuda \
  --checkpoint checkpoints/toy_condot_velocity.pt \
  --batch-size 1024 --steps 30 --repeats 100 --schedule uniform

python benchmark/benchmark_vp.py --device cuda \
  --checkpoint checkpoints/toy_vpsde_score.pt \
  --batch-size 1024 --steps 30 --repeats 100 --schedule logsnr
```

一键运行全部模型、求解器、统计和可视化：

```bash
python run_all.py --device cuda --batch-size 1024 --steps 30 \
  --repeats 100 --reference-steps 512 --output-dir outputs/all_aligned
```

输出目录中的文件含义：

| 文件 | 说明 |
|---|---|
| `results.csv` | 原始统计数据，可供二次绘图/分析 |
| `results.md` | 文本结果 |
| `table.png` | 静态统计表图片 |
| `summary.png` | 延迟、数值误差与分布质量汇总图 |
| `samples.png` | 同一初始噪声和条件下的样本散点图 |

## 6. 指标应该怎样解释

| 指标 | 含义 | 可比较范围 |
|---|---|---|
| `NFE` | decoder 实际调用次数 | 所有方法，决定主要推理成本 |
| `latency_ms` | 同设备、同 batch 下的平均延迟 | 所有方法 |
| `reference_mse` | 到同一 checkpoint 的 RK4(512) 轨迹的 MSE | **仅同一 prediction 行内**的数值积分误差 |
| `mean_error` | 条件样本均值到目标 mode 均值的误差 | 可跨模型 |
| `covariance_error` | 生成协方差到 $0.25^2I$ 的误差 | 可跨模型 |
| `w2` | 条件拟合高斯到真实高斯的 Wasserstein-2 距离 | 主要分布质量指标，越小越好 |
| `component_std` | 每个 mode 的平均标准差 | 理想值为 $0.25$ |
| `finite_fraction` | 有限数值样本占比 | 小于 1 表示 NaN/Inf；等于 1 不代表没有大幅爆炸 |

`reference_mse` 不能拿来比较不同网络的模型质量，因为每个 checkpoint 都有其自身的学习 ODE 和 RK4 参考轨迹。跨参数化应看 W2、均值/协方差误差及 `samples.png`。

## 7. 当前实验与结论

以下结果来自 `outputs/all_aligned/`：四条件二维多峰高斯、batch size 1024、30 个积分区间、100 次重复计时、每个 checkpoint 用 RK4(512) 作自身数值参考。重新训练或更换硬件后的绝对数值会有波动。

### 7.1 各输出类型的最佳 W2 组合

| 概率路径 | 网络输出 | 最佳求解器 | W2 ↓ | NFE | 平均延迟 |
|---|---|---|---:|---:|---:|
| CondOT | velocity | Adams--Bashforth 2 | 0.04340 | 30 | 73.60 ms |
| CondOT | score | Adams--Bashforth 2 | 0.08824 | 30 | 74.14 ms |
| CondOT | data | 显式欧拉法 | 0.14824 | 30 | 73.49 ms |
| VP-SDE | noise | DPM-Solver 2M | 0.06260 | 31 | 80.28 ms |
| VP-SDE | x_start | 显式欧拉法 | 0.06053 | 30 | 74.40 ms |
| VP-SDE | pf_velocity | Adams--Bashforth 2 | 0.07322 | 30 | 73.44 ms |
| VP-SDE | score | DPM-Solver++ 2M | 0.05711 | 31 | 80.70 ms |

### 7.2 工程结论

1. **CondOT 的直接 velocity 预测最可靠。** `velocity + AB2` 取得当前总表最佳 W2（0.04340），仅需 30 NFE。显式二阶法和 RK4 的质量接近，说明该模型的速度场足够平滑。
2. **固定 decoder 预算下，AB2 是 CondOT 的默认候选。** 它只保存一次历史速度，而 midpoint/Heun 需要 60 NFE，RK4 需要 120 NFE。
3. **CondOT 的 direct-data 不是天然更适合的表示。** $1/(1-t)$ 会放大末端的 data 误差；当前小模型中即便采用速度等价加权，data 仍弱于直接 velocity。
4. **VP-SDE 的输出必须与 DPM 方法匹配。** `score + DPM-Solver++ 2M` 是 VP 内最佳组合；`noise + DPM-Solver 2M` 也稳定。前者适合 data-oriented 的半解析公式，后者适合 noise/score 表示。
5. **PF ODE velocity 可稳定积分，却不必然是最易学习的输出。** `pf_velocity + AB2` 正常工作；但其转换为 $x_0$ 在高噪声端病态，不能直接搭配 DPM-Solver++。
6. **不能只按 RK4-reference MSE 选方法。** RK4 误差最低是自然结果，但耗时/NFE 大约为 AB2 的四倍。部署时应先设定 NFE 或延迟预算，再按 W2 或真实下游指标选择。

### 7.3 已知不兼容组合：PF velocity + DPM-Solver++ 2M

`outputs/all_aligned/vpsde/samples.png` 中 `pf_velocity` 最后一个 DPM-Solver++ 2M 面板为空，并非绘图遗漏：输出虽是有限数，但已远超绘图范围；该组合约有

$$
\mathrm{W2}\approx282,
\qquad \mathrm{component\_std}\approx192.
$$

由 PF velocity 到 data prediction 的误差传播为

$$
\epsilon_\theta=\sigma_t\left(x_t+\frac{2u_\theta}{\beta(t)}\right),
\qquad
\delta\hat{x}_0=-\frac{2\sigma_t^2}{\alpha_t\beta(t)}\delta u.
$$

采样初段 $α_t\to0$，因此很小的 velocity 预测误差会在 $x_0$ 表示中被极大放大。DPM-Solver++ 2M 的半解析积分依赖稳定的 clean-data prediction，继而多步外推失败。这不是 probability-flow ODE 必然发散，而是“输出表示 + 求解器”不匹配。

建议：PF velocity 使用 `euler`、`ab2`、`midpoint`、`heun` 或 `rk4`；若需要 DPM-Solver++，直接训练 `x_start`/data 头，或使用 `(pf_velocity, x_start)` 双头模型，而不是从 velocity 间接恢复 $x_0$。

## 8. 推荐选择

| 场景 | 推荐输出 | 推荐求解器 | 原因 |
|---|---|---|---|
| CondOT / Flow Matching，质量与速度兼顾 | `velocity` | `ab2` | 当前最优成本/质量折中 |
| CondOT，验证数值精度 | `velocity` | `midpoint`、`heun` 或 `rk4` | 更多 NFE 换更低积分误差 |
| VP-SDE，使用 DPM-Solver++ | `score` 或 `x_start` | `dpmsolverpp_2m` | 与 data-oriented 结构匹配 |
| VP-SDE，使用 DPM-Solver | `noise` 或 `score` | `dpmsolver_2m` | 与 noise/score 表示匹配 |
| VP-SDE，直接预测 PF drift | `pf_velocity` | `ab2` | 避免病态恢复 $x_0$ |

迁移到真实规划系统时，请固定条件、初始噪声、NFE 和时间网格，并同时报告碰撞率、舒适性、进度、多样性等下游指标；toy W2 只能作为求解器和参数化的诊断指标。

## 9. 快速自检

```bash
python tests/test_solvers.py
```

测试覆盖参数化转换、基本 NFE/阶数约束与非均匀时间网格上的 AB2 系数。
