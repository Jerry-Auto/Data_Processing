#ifndef LQR_H
#define LQR_H

#include <casadi/casadi.hpp>
#include <vector>

using namespace casadi;

// 声明动力学函数构造方法
Function build_dynamics(double dt);

// 声明代价函数构造方法
Function build_cost(const DM& Q, const DM& R, const DM& Qf);

// 直接法 Demo 函数声明
void direct_method_demo();

// 迭代法 Demo 函数声明
void iterative_method_demo();

void cilqr_demo();

DM compute_trajectory_cost(
    const std::vector<DM>& X, 
    const std::vector<DM>& U, 
    const DM& Q, 
    const DM& R, 
    const DM& Qf, 
    const DM& x_goal);

#endif // LQR_H