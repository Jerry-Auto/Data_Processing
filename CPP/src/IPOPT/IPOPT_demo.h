#pragma once

#include <iostream>
#include <cppad/ipopt/solve.hpp>



// 定义目标函数和约束条件的类
class FG_eval
{
public:
  // 定义AD（自动微分）向量类型
  typedef CPPAD_TESTVECTOR(CppAD::AD<double>) ADvector;
  // 重载()运算符，定义优化问题的目标函数和约束条件
  void operator()(ADvector& fg, const ADvector& x);
};

bool get_started(void);