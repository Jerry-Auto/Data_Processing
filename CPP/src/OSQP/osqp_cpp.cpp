#include <OsqpEigen/OsqpEigen.h>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <iostream>
#include <string>

// 修正后的状态转换函数
std::string statusToString(OsqpEigen::Status status) {
    switch (status) {
        case OsqpEigen::Status::Solved:                 return "Solved"; // 对应之前的 Optimal
        case OsqpEigen::Status::SolvedInaccurate:      return "SolvedInaccurate";
        case OsqpEigen::Status::MaxIterReached:        return "MaxIterReached"; // 对应之前的 MaxIterations
        case OsqpEigen::Status::PrimalInfeasible:      return "PrimalInfeasible";
        case OsqpEigen::Status::PrimalInfeasibleInaccurate: return "PrimalInfeasibleInaccurate";
        case OsqpEigen::Status::DualInfeasible:        return "DualInfeasible";
        case OsqpEigen::Status::DualInfeasibleInaccurate:  return "DualInfeasibleInaccurate";
        case OsqpEigen::Status::NonCvx:                return "NonCvx"; // 对应 NonConvex
        // 根据你的需要添加其他状态
        default: return "Unknown Status: " + std::to_string(static_cast<int>(status));
    }
}


int main() {
    // 1. 定义问题维度
    int num_variables = 2;
    int num_constraints = 3;

    // 2. 初始化问题数据（保持不变）
    Eigen::SparseMatrix<double> P(num_variables, num_variables);
    P.insert(0, 0) = 4.0;
    P.insert(0, 1) = 1.0;
    P.insert(1, 0) = 1.0;
    P.insert(1, 1) = 2.0;

    Eigen::VectorXd q(num_variables);
    q << 1.0, 1.0;

    Eigen::SparseMatrix<double> A(num_constraints, num_variables);
    A.insert(0, 0) = 1.0;
    A.insert(0, 1) = 1.0;
    A.insert(1, 0) = 1.0;
    A.insert(2, 1) = 1.0;

    Eigen::VectorXd l(num_constraints);
    Eigen::VectorXd u(num_constraints);
    l << 1.0, 0.0, 0.0;
    u << 1.0, 0.7, 0.7;

    // 3. 实例化求解器
    OsqpEigen::Solver solver;

    // 4. 配置设置
    solver.settings()->setVerbosity(true);
    solver.settings()->setWarmStart(true);

    // 5. 设置问题数据
    solver.data()->setNumberOfVariables(num_variables);
    solver.data()->setNumberOfConstraints(num_constraints);

    solver.data()->setHessianMatrix(P);
    solver.data()->setGradient(q);
    solver.data()->setLinearConstraintsMatrix(A);
    solver.data()->setLowerBound(l);
    solver.data()->setUpperBound(u);

    solver.initSolver();

    // 修正后的求解和状态检查逻辑
    OsqpEigen::ErrorExitFlag exitFlag = solver.solveProblem(); // 返回的是 ErrorExitFlag 类型

    // 检查求解过程是否出错（比如数据或设置问题）
    if (exitFlag != OsqpEigen::ErrorExitFlag::NoError) {
        std::cerr << "求解过程发生错误，错误码: " << static_cast<int>(exitFlag) << std::endl;
        // 这里可以根据具体的 ErrorExitFlag 进行更细致的错误处理
        return 1;
    }

    // 获取并输出求解状态
    OsqpEigen::Status solutionStatus = solver.getStatus();
    std::cout << "求解状态: " << statusToString(solutionStatus) << std::endl;

    // 根据状态判断是否成功
    if (solutionStatus == OsqpEigen::Status::Solved || solutionStatus == OsqpEigen::Status::SolvedInaccurate) {
        // 成功求解，获取解
        Eigen::VectorXd solution = solver.getSolution();
        std::cout << "求解成功！" << std::endl;
        std::cout << "目标函数值: " << solver.getObjValue() << std::endl;
        std::cout << "解为: \n" << solution << std::endl;
    } else {
        // 处理未成功求解的情况
        std::cout << "求解未成功。" << std::endl;
    }

    return 0;
}