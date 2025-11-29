#pragma once

#include <vector>           // 标准向量容器库
#include <cppad/cppad.hpp>  // CppAD自动微分库
namespace
{  // 开始空的匿名命名空间
  
/**
 * 多项式函数模板
 * 功能：计算多项式 a[0] + a[1]*x + a[2]*x² + ... + a[k-1]*x^(k-1) 的值
 * 模板参数：Type - 可以是普通double类型或CppAD的自动微分类型
 * 参数：a - 多项式系数向量，x - 自变量
 * 返回值：多项式在x处的函数值
 */
template <class Type>
Type Poly(const CPPAD_TESTVECTOR(double) & a, const Type& x)
{
  size_t k = a.size();  // 获取多项式系数的个数
  Type y = 0.;          // 初始化求和结果为0
  Type x_i = 1.;        // 初始化x的幂次，从x^0 = 1开始
  
  // 循环计算多项式的每一项并累加
  for (size_t i = 0; i < k; i++)
  {
    y += a[i] * x_i;  // 累加当前项：系数a[i] × x的i次幂
    x_i *= x;         // 计算x的下一个幂次：x_i = x_i × x
  }
  return y;  // 返回多项式计算结果
}
  
}  // 匿名命名空间结束


/* 计算向量输入，向量输出的示例 */
int vector_demo();

/**
 * CppAD自动微分库示例程序
 * 功能：计算多项式函数 f(x) = 1 + x + x² + x³ + x⁴ 在 x=3 处的导数
 */
int scalar_demo();

int higher_order_demo();

int auto_sparse_demo();

bool simple_sparse_jacobian(void);

int sparse_jac_demo();

int multi_output_sparse_hessian() ;