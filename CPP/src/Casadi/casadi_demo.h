#pragma once

#include <casadi/casadi.hpp>

namespace casadi_demo {

// Solve a tiny constrained NLP with IPOPT:
//   min x^2 + 100 z^2
//   s.t. z + (1-x)^2 - y = 0
// Returns the optimal decision vector [x,y,z].
// 中文：用 IPOPT 解一个最小可跑的约束 NLP。
// - 决策变量：(x, y, z)
// - 目标函数：x^2 + 100 z^2
// - 等式约束：z + (1-x)^2 - y = 0
// 返回：最优解向量 [x,y,z]（DM 类型）。
casadi::DM solve_rosenbrock_nlp();

// Demonstrate Jacobian/Hessian construction (symbolic) and printing.
// 中文：演示 CasADi 的自动求导接口：Jacobian / Hessian（符号层面构造并打印）。
void demo_derivatives();

// Demonstrate basic CasADi C++ usage patterns: core types, Function creation/call,
// slicing, and matrix operations.
// 中文：补充一些最常用的基础用法示例：
// - SX/MX/DM 的创建与打印
// - Function 的构造（带输入/输出命名）
// - Function 的两种调用方式：vector<DM> 与 DMDict
// - Slice/reshape/矩阵乘法（mtimes）
void demo_basic_usage();

// Demonstrate ODE integration: x_dot = -p*x from t0=0 to tf=1.
// Returns xf as DM scalar.
// 中文：演示 ODE 积分器 integrator。
// - 模型：x_dot = -p * x
// - 时间区间：[0, 1]
// - 说明：优先尝试 cvodes 插件，不可用则 fallback 到 rk。
// 返回：终值 xf（DM 标量）。
casadi::DM demo_integrator(double p, double x0);

// Demonstrate solving a tiny convex QP with qpsol (tries multiple plugins).
// 中文：演示 QP 求解（qpsol）。
// - 不同安装的 QP 插件可用性不同：会尝试 qpoases / osqp / qrqp。
void demo_qp();

// Demonstrate rootfinder on a small nonlinear equation.
// 中文：演示 rootfinder（非线性方程求根），这里用 newton 方法求 sqrt(2)。
void demo_rootfinder();

// Demonstrate Opti (tries ipopt, falls back to sqpmethod).
// 中文：演示 Opti 建模接口：
// - 优先用 ipopt 求解
// - 若 ipopt 不可用，fallback 到 sqpmethod（内部 QP 默认用 qrqp）。
void demo_opti();


}  // namespace casadi_demo
