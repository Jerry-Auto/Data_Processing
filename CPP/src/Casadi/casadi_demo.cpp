#include "casadi_demo.h"

#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace casadi_demo {

casadi::DM solve_rosenbrock_nlp() {
  using casadi::DM;
  using casadi::Function;
  using casadi::SX;

  // 中文：SX 适合构造小型符号表达式（这里是标量/小维度）。
  SX x = SX::sym("x");
  SX y = SX::sym("y");
  SX z = SX::sym("z");

  const SX two = SX(2);

  casadi::SXDict nlp{
      {"x", vertcat(std::vector<SX>{x, y, z})},
      {"f", pow(x, two) + 100 * pow(z, two)},
      {"g", z + pow(1 - x, two) - y},
  };

  // 中文：nlpsol(name, solver_plugin, nlp_dict)
  // - name: solver 名称（用于内部标识/打印等）
  // - solver_plugin: "ipopt" 等
  // - nlp_dict: {"x","f","g"}
  Function solver = casadi::nlpsol("S", "ipopt", nlp);

  casadi::DMDict arg;
  // 中文：初值/约束上下界。
  arg["x0"] = DM(std::vector<double>{2.5, 3.0, 0.75});
  arg["lbg"] = DM(0);
  arg["ubg"] = DM(0);

  casadi::DMDict res = solver(arg);
  return res.at("x");
}

void demo_derivatives() {
  using casadi::SX;

  // 中文：构造一个向量函数 f(x0,x1) 并对输入 x=[x0,x1] 求导。
  SX x0 = SX::sym("x0");
  SX x1 = SX::sym("x1");
  SX x = vertcat(std::vector<SX>{x0, x1});

  const SX two = SX(2);
  SX f = vertcat(std::vector<SX>{
      pow(x0, two) + sin(x1),
      x0 * x1,
  });

  // 中文：Jacobian: J = df/dx
  SX J = jacobian(f, x);
  SX g;
  // 中文：Hessian: H = d^2(dot(f,f))/dx^2，同时返回 gradient g
  SX H = hessian(dot(f, f), x, g);

  std::cout << "f(x) = " << f << "\n";
  std::cout << "J = " << J << "\n";
  std::cout << "grad = " << g << "\n";
  std::cout << "H = " << H << "\n";
}

void demo_basic_usage() {
  using casadi::DM;
  using casadi::Function;
  using casadi::MX;
  using casadi::Slice;
  using casadi::SX;

  std::cout << "--- 基础用法：核心类型 / Function / 调用方式" << "\n";

  // 1) SX / MX / DM：创建与打印
  // 中文：SX/MX 是符号类型（表达式图），DM 是数值矩阵（用于输入/输出/初值/解）。
  SX xs = SX::sym("xs");
  MX xm = MX::sym("xm", 2, 2);
  DM d = DM(std::vector<double>{1, 2, 3, 4});
  DM d2 = DM::reshape(d, 2, 2);
  std::cout << "SX xs = " << xs << "\n";
  std::cout << "MX xm = " << xm << "\n";
  std::cout << "DM reshape([1,2,3,4],2,2) = " << d2 << "\n";

  // 2) Function：封装表达式（建议给输入/输出命名，方便用 DMDict 调用）
  // 定义：f(x,y) = [ x0+y , sin(y)*x1 ]
  MX x = MX::sym("x", 2);
  MX y = MX::sym("y");
  MX out0 = x(0) + y;
  // 中文：这版 CasADi 的很多数学函数/mtimes 是通过 ADL（友元函数）暴露的，
  // 用法通常是直接写 sin(y)、mtimes(A,B)，不要强行写 casadi::sin/casadi::mtimes。
  MX out1 = sin(y) * x(1);
  Function f("f", {x, y}, {out0, out1}, {"x", "y"}, {"out0", "out1"});

  // 2.1) 调用方式 A：按位置传 vector<DM>
  // 中文：适合快速调用，但可读性略差（要记住输入顺序）。
  std::vector<DM> out_vec = f(std::vector<DM>{DM(std::vector<double>{1, 2}), DM(0.5)});
  std::cout << "call by vector<DM>: out0=" << out_vec.at(0) << ", out1=" << out_vec.at(1) << "\n";

  // 2.2) 调用方式 B：按名字传 DMDict（更推荐）
  // 中文：更清晰，尤其是输入/输出很多的时候。
  casadi::DMDict arg;
  arg["x"] = DM(std::vector<double>{1, 2});
  arg["y"] = DM(0.5);
  casadi::DMDict res = f(arg);
  std::cout << "call by DMDict: out0=" << res.at("out0") << ", out1=" << res.at("out1") << "\n";

  // 3) Slice / reshape / mtimes
  // 中文：C++ 没有 Python 的 [:] 语法糖，CasADi 用 Slice。
  // - Slice all;  表示“全选”
  // - Slice(start, stop, step)  表示区间切片（0 基索引；stop 不包含；step 默认 1）
  //
  // 另外注意：DM::reshape 的填充是“列优先”（column-major）风格：
  // 例如 [1,2,3,4] reshape 成 2x2，会得到 [[1,3],[2,4]]（你前面的输出就是这样）。
  DM M = DM::reshape(DM(std::vector<double>{1, 2, 3, 4, 5, 6}), 2, 3);  // 2x3
  Slice all;
  std::cout << "M = " << M << "\n";

  // 3.1 取一整行/一整列（读切片）
  // - M(0, all)  第 0 行（1x3）
  // - M(all, 1)  第 1 列（2x1）
  DM row0 = M(0, all);
  DM col1 = M(all, 1);
  std::cout << "M(0,:) = " << row0 << "\n";
  std::cout << "M(:,1) = " << col1 << "\n";

  // 3.2 子块切片：取连续区间的列/行
  // 这里取列 [1, 3) -> 第 1、2 列（2x2）
  DM cols_1_3 = M(all, Slice(1, 3));
  std::cout << "M(:,1:3) = " << cols_1_3 << "\n";

  // 3.3 步长切片：Slice(start, stop, step)
  // 这里取列 [0, 3) 步长 2 -> 第 0、2 列（2x2）
  DM cols_step = M(all, Slice(0, 3, 2));
  std::cout << "M(:,0:3:2) = " << cols_step << "\n";

  // 备注：对 DM/MX/SX，切片本质上会构造一个“子表达/子矩阵”。
  // 对符号变量（MX/SX）尤其常用；对 DM 这里主要用于演示与调试。

  // 中文：矩阵乘法用 mtimes；`A*B` 在 CasADi 里是逐元素（element-wise）。
  DM A = DM::reshape(DM(std::vector<double>{1, 2, 3, 4}), 2, 2);
  DM B = DM::reshape(DM(std::vector<double>{5, 6, 7, 8}), 2, 2);
  DM C = mtimes(A, B);
  std::cout << "A = " << A << "\n";
  std::cout << "B = " << B << "\n";
  std::cout << "mtimes(A,B) = " << C << "\n";
}

casadi::DM demo_integrator(double p_val, double x0_val) {
  using casadi::DM;
  using casadi::Function;
  using casadi::SX;

  // 中文：符号化 ODE：x_dot = -p*x
  SX x = SX::sym("x");
  SX p = SX::sym("p");

  casadi::SXDict dae{{"x", x}, {"p", p}, {"ode", -p * x}};

  Function F;
  try {
    // 中文：cvodes 通常更稳健/更通用，但需要对应插件可用。
    F = casadi::integrator("F", "cvodes", dae, 0.0, 1.0);
  } catch (const std::exception&) {
    // 中文：若 cvodes 插件不可用，退化到固定步长 RK。
    F = casadi::integrator("F", "rk", dae, 0.0, 1.0);
  }

  casadi::DMDict arg;
  arg["x0"] = DM(x0_val);
  arg["p"] = DM(p_val);

  casadi::DMDict res = F(arg);
  return res.at("xf");
}

void demo_qp() {
  using casadi::DM;
  using casadi::Dict;
  using casadi::Function;
  using casadi::SX;

  // 中文：QP 形式（CasADi 会自动识别为二次目标+线性约束的结构）
  //   min (x0-1)^2 + (x1-2)^2
  //   s.t. x0 + x1 = 1
  //   并在求解参数里加 x>=0 的边界。
  SX x = SX::sym("x", 2);
  const SX one = SX(1);
  const SX two = SX(2);
  SX f = pow(x(0) - one, two) + pow(x(1) - two, two);
  SX g = x(0) + x(1);

  casadi::SXDict qp{{"x", x}, {"f", f}, {"g", g}};

  Function solver;
  std::string used_plugin;
  // 中文：QP 插件在不同系统/安装方式下可能不存在，所以按顺序尝试。
  for (const std::string& plugin : std::vector<std::string>{"qpoases", "osqp", "qrqp"}) {
    try {
      Dict opts;
      if (plugin == "qpoases") {
        // 中文：qpoases 的常用选项
        opts["sparse"] = true;
        opts["print_time"] = false;
      }
      solver = casadi::qpsol("solver", plugin, qp, opts);
      used_plugin = plugin;
      break;
    } catch (const std::exception&) {
      // Try next plugin
    }
  }

  if (used_plugin.empty()) {
    std::cout << "QP 演示已跳过：未找到可用的 qpsol 插件（尝试了 qpoases/osqp/qrqp）。" << "\n";
    return;
  }

  casadi::DMDict arg;
  // 中文：变量边界 x >= 0
  arg["lbx"] = DM(std::vector<double>{0.0, 0.0});
  arg["ubx"] = DM(std::vector<double>{casadi::inf, casadi::inf});
  // 中文：等式约束 g=1
  arg["lbg"] = DM(1.0);
  arg["ubg"] = DM(1.0);

  casadi::DMDict res = solver(arg);
  std::cout << "QP 使用插件：" << used_plugin << "\n";
  std::cout << "最优 x = " << res.at("x") << "\n";
  std::cout << "最优 f = " << res.at("f") << "\n";
}

void demo_rootfinder() {
  using casadi::DM;
  using casadi::Dict;
  using casadi::Function;
  using casadi::SX;

  // Solve v^2 - a = 0 for v, given a
  // 中文：构造 residual r(v,a)=v^2-a，rootfinder 会求 r=0 的解。
  SX v = SX::sym("v");
  SX a = SX::sym("a");
  Function vfcn("vfcn", {v, a}, {v * v - a});

  try {
    Dict opts;
    // 中文：rootfinder(name, solver_plugin, residual_function, opts)
    // 这里用 newton；输入按 {初值, 参数...} 传入。
    Function ifcn = casadi::rootfinder("ifcn", "newton", vfcn, opts);
    DM v0 = DM(1.0);
    DM aval = DM(2.0);
    DM sol = ifcn(std::vector<DM>{v0, aval}).at(0);
    std::cout << "rootfinder（newton）：sqrt(2) ≈ " << sol << "\n";
  } catch (const std::exception& e) {
    std::cout << "Rootfinder 演示已跳过：" << e.what() << "\n";
  }
}

void demo_opti() {
  using casadi::DM;

  // Simple bound-constrained problem: min (x-3)^2 s.t. 1<=x<=5
  // 中文：Opti 是更“贴近数学表达”的建模接口。
  // 这里解：min (x-3)^2，约束 1<=x<=5。
  casadi::Opti opti;
  casadi::MX x = opti.variable();
  // 中文：MX 上建议用 sq() 这类 CasADi 原生函数，避免被解析成 std::pow。
  opti.minimize(casadi::sq(x - 3));
  opti.subject_to(x >= 1);
  opti.subject_to(x <= 5);

  try {
    // 中文：优先尝试 ipopt（如果你的 CasADi 安装带了 ipopt 插件）。
    opti.solver("ipopt", {{}}, {{"print_time", false}, {"ipopt.print_level", 0}});
    casadi::OptiSol sol = opti.solve();
    std::cout << "Opti(ipopt): x* = " << sol.value(x) << "\n";
    return;
  } catch (const std::exception&) {
    // Fall back
  }

  try {
    // 中文：fallback 到 sqpmethod（序列二次规划），内部需要一个 QP 求解器。
    casadi::Dict opts;
    opts["qpsol"] = "qrqp";
    opti.solver("sqpmethod", opts);
    casadi::OptiSol sol = opti.solve();
    std::cout << "Opti(sqpmethod): x* = " << sol.value(x) << "\n";
  } catch (const std::exception& e) {
    std::cout << "Opti 演示已跳过：" << e.what() << "\n";
  }
}


}  // namespace casadi_demo

int main() {
  try {
    std::cout << "[1] 求解 NLP（ipopt）" << "\n";
    casadi::DM x_opt = casadi_demo::solve_rosenbrock_nlp();
    std::cout << "最优 x = " << x_opt << "\n\n";

    std::cout << "[2] 基础用法演示" << "\n";
    casadi_demo::demo_basic_usage();
    std::cout << "\n";

    std::cout << "[3] 求导示例" << "\n";
    casadi_demo::demo_derivatives();
    std::cout << "\n";

    std::cout << "[4] 积分器示例" << "\n";
    // 中文：不同 p 下的指数衰减终值。
    for (double p : {0.5, 1.0, 2.0}) {
      casadi::DM xf = casadi_demo::demo_integrator(p, 1.0);
      std::cout << "p=" << p << "  x(1)=" << xf << "\n";
    }

    std::cout << "\n[5] 二次规划示例（qpsol）" << "\n";
    casadi_demo::demo_qp();

    std::cout << "\n[6] 求根器示例" << "\n";
    casadi_demo::demo_rootfinder();

    std::cout << "\n[7] Opti 示例" << "\n";
    casadi_demo::demo_opti();

    std::cout << "\n完成。" << "\n";
    return 0; 
  } catch (const std::exception& e) {
    std::cerr << "错误: " << e.what() << "\n";
    std::cerr << "\n注意：C++ 演示需要在编译/链接时可用的 CasADi C++ 头文件和库。" << "\n";
    std::cerr << "如果你仅通过 pip 安装了 Python casadi，C++ 链接库可能并不可用。" << "\n";
    return 1;
  }
}
