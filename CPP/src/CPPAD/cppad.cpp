#include "AD_demo.h"

/**
 * 主函数 - 演示CppAD自动微分的完整流程
 */
int main(void)
{
  int out=0;
  // out=out||scalar_demo();
  out=out||higher_order_demo();
  // out=out||vector_demo();
  // out=out||auto_sparse_demo();
  // simple_sparse_jacobian();
  // sparse_jac_demo();
  // out=out||multi_output_sparse_hessian();
  return out;
}