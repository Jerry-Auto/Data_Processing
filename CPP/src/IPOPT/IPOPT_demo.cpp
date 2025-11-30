#include"IPOPT_demo.h"
using CppAD::AD;
using namespace std;

// 重载()运算符，定义优化问题的目标函数和约束条件
void FG_eval::operator()(ADvector& fg, const ADvector& x)
{
    assert(fg.size() == 3);  // fg[0]是目标函数，fg[1-2]是约束条件
    assert(x.size() == 4);   // 有4个优化变量

    // 提取优化变量
    AD<double> x1 = x[0];
    AD<double> x2 = x[1];
    AD<double> x3 = x[2];
    AD<double> x4 = x[3];

    // fg[0] = 目标函数: x1*x4*(x1+x2+x3) + x3
    fg[0] = x1 * x4 * (x1 + x2 + x3) + x3;

    // fg[1] = 第一个约束条件: x1*x2*x3*x4 (不等式约束)
    fg[1] = x1 * x2 * x3 * x4;

    // fg[2] = 第二个约束条件: x1² + x2² + x3² + x4² (等式约束)
    fg[2] = x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4;

    return;
}

// 主要的优化求解函数
bool get_started(void)
{
  bool ok = true;      // 用于检查求解是否成功
  size_t i;            // 循环索引
  typedef CPPAD_TESTVECTOR(double) Dvector;  // 定义双精度向量类型

  // 问题规模定义
  size_t nx = 4;   // 优化变量的个数
  size_t ng = 2;   // 约束条件的个数
  
  // 设置初始点
  Dvector x0(nx);
  x0[0] = 1.0;  // x1初始值
  x0[1] = 5.0;  // x2初始值  
  x0[2] = 5.0;  // x3初始值
  x0[3] = 1.0;  // x4初始值

  // 设置变量的下界和上界 [1.0, 5.0]
  Dvector xl(nx), xu(nx);
  for (i = 0; i < nx; i++)
  {
    xl[i] = 1.0;  // 所有变量的下界
    xu[i] = 5.0;  // 所有变量的上界
  }

  // 设置约束条件的下界和上界
  Dvector gl(ng), gu(ng);
  // 第一个约束: x1*x2*x3*x4 ≥ 25 (不等式约束)
  gl[0] = 25.0;      // 下界
  gu[0] = 1.0e19;    // 上界设为很大的数，表示只有下界约束
  
  // 第二个约束: x1² + x2² + x3² + x4² = 40 (等式约束)
  gl[1] = 40.0;      // 下界
  gu[1] = 40.0;      // 上界等于下界，表示等式约束

  // 创建目标函数和约束条件的评估对象
  FG_eval fg_eval;

  // 设置IPOPT求解器的选项
  string options;
  // 关闭输出打印 (0=无输出, 5=详细输出)
  options += "Integer print_level  0\n";
  // 隐藏求解器横幅
  options += "String sb            yes\n";
  // 设置最大迭代次数
  options += "Integer max_iter     10\n";
  // 设置收敛容差
  options += "Numeric tol          1e-6\n";
  // 设置导数检验方式（二阶导数检验）
  options += "String derivative_test   second-order\n";
  // 设置扰动半径为0（不使用随机扰动）
  options += "Numeric point_perturbation_radius   0.\n";

  // 声明求解结果对象
  CppAD::ipopt::solve_result<Dvector> solution;
  
  // 调用IPOPT求解器求解优化问题
  // 参数说明：
  // - options: 求解器选项
  // - x0: 初始点
  // - xl, xu: 变量边界
  // - gl, gu: 约束边界  
  // - fg_eval: 目标函数和约束评估对象
  // - solution: 求解结果
  CppAD::ipopt::solve<Dvector, FG_eval>(options, x0, xl, xu, gl, gu, fg_eval, solution);

  // 输出求解结果
  cout << "优化解: " << solution.x << endl;
  cout << "目标函数值: " << solution.obj_value << endl;
  cout << "求解状态: " << solution.status << endl;

  // 检查求解是否成功
  ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

  // 预期的解（用于验证）
  double check_x[] = { 1.000000, 4.743000, 3.82115, 1.379408 };
  double check_zl[] = { 1.087871, 0., 0., 0. };
  double check_zu[] = { 0., 0., 0., 0. };
  
  // 设置数值比较的容差
  double rel_tol = 1e-6;  // 相对容差
  double abs_tol = 1e-6;  // 绝对容差
  
  // 验证解的准确性
  for (i = 0; i < nx; i++)
  {
    ok &= CppAD::NearEqual(check_x[i], solution.x[i], rel_tol, abs_tol);
    ok &= CppAD::NearEqual(check_zl[i], solution.zl[i], rel_tol, abs_tol);
    ok &= CppAD::NearEqual(check_zu[i], solution.zu[i], rel_tol, abs_tol);
  }

  // 输出验证结果
  if (ok) {
    cout << "求解成功且结果验证通过!" << endl;
  } else {
    cout << "求解失败或结果验证未通过!" << endl;
  }

  return ok;
}
