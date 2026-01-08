#include "LQR.h"
#include <iostream>
#include <vector>

using namespace casadi;

// === 动态调整 mu 的范围，避免过大或过小 ===
// mu = std::clamp(mu, 1e-12, 1e6);

// === 修改 build_dynamics 以支持数值化输出 ===
Function build_dynamics(double dt, int nx, int nu) {
    MX x = MX::sym("x", nx, 1);  // 状态变量
    MX u = MX::sym("u", nu, 1);  // 控制变量
    // 约定状态顺序为 [p_x, p_y, theta, v]
    MX f_cont = MX::vertcat({x(3) * cos(x(2)), x(3) * sin(x(2)), u(1), u(0)});
    MX x_next = x + dt * f_cont;
    MX Fx = jacobian(x_next, x);
    MX Fu = jacobian(x_next, u);
    return Function("Fdyn", {x, u}, {x_next, Fx, Fu}, {"x", "u"}, {"x_next", "Fx", "Fu"});
}

// === 修复 build_cost 函数 ===
// 确保所有变量被正确绑定到 Function 的输入或输出
Function build_cost(const DM& Q, const DM& R, int nx, int nu) {
    MX x = MX::sym("x", nx, 1);
    MX u = MX::sym("u", nu, 1);
    MX xr = MX::sym("xr", nx, 1);

    MX diff = x - xr;
    MX l = mtimes(mtimes(diff.T(), Q), diff) + mtimes(mtimes(u.T(), R), u);
    MX z = MX::vertcat({x, u});
    MX gz;
    MX Hfull = hessian(l, z, gz);

    // 显式绑定所有输出变量（使用传入的维度）
    MX lx = gz(Slice(0, nx), 0);
    MX lu = gz(Slice(nx, nx + nu), 0);
    MX lxx = Hfull(Slice(0, nx), Slice(0, nx));
    MX luu = Hfull(Slice(nx, nx + nu), Slice(nx, nx + nu));
    MX lux = Hfull(Slice(nx, nx + nu), Slice(0, nx));

    return Function("cost_der", {x, u, xr}, {lx, lu, lxx, luu, lux},
                    {"x", "u", "xr"}, {"lx", "lu", "lxx", "luu", "lux"});
}

// === 构建不等式约束函数 ===
Function build_inequality_constraints(const DM& u_min, const DM& u_max) {
    MX u = MX::sym("u", 2, 1);  // 控制变量 [steering_angle, acceleration]

    // 不等式约束: u_min <= u <= u_max
    MX g_lower = u_min - u; // u >= u_min  =>  u_min - u <= 0
    MX g_upper = u - u_max; // u <= u_max  =>  u - u_max <= 0

    // 返回约束函数
    return Function("IneqConstr", {u}, {vertcat(g_lower, g_upper)}, {"u"}, {"g"});
}

// === 构建状态不等式约束函数 ===
Function build_state_constraints(const DM& x_min, const DM& x_max) {
    MX x = MX::sym("x", 4, 1);  // 状态变量 [x_pos, y_pos, yaw, v]

    // 不等式约束: x_min <= x <= x_max
    MX g_lower = x_min - x; // x >= x_min  =>  x_min - x <= 0
    MX g_upper = x - x_max; // x <= x_max  =>  x - x_max <= 0

    // 返回约束函数
    return Function("StateConstr", {x}, {vertcat(g_lower, g_upper)}, {"x"}, {"g"});
}

// === 直接法 Demo 实现 ===
void direct_method_demo() {
    // 参数设置
    const int N = 20; // 时间步数
    const double dt = 0.1; // 时间步长
    const int state_dim = 4; // 状态维度 [x_pos, y_pos, yaw, v]
    const int control_dim = 2; // 控制维度 [steering_angle, acceleration]

    // 状态代价矩阵
    const DM Q = DM::diag({10, 10, 1, 1}); // 状态代价权重
    const DM R = DM::diag({1, 1});         // 控制代价权重
    const DM Qf = DM::diag({10, 10, 1, 1}); // 终端状态代价权重

    // 构造动力学和代价函数（传入维度信息）
    Function dynamics = build_dynamics(dt, state_dim, control_dim);
    Function cost = build_cost(Q, R, state_dim, control_dim);

    // 定义符号变量，名称只起标识作用，不会被求解器识别，方便读者看
    MX x = MX::sym("x", state_dim, N + 1); // 状态变量
    MX u = MX::sym("u", control_dim, N);   // 控制变量
    MX x0 = MX::sym("x0", state_dim);      // 初始状态
    MX x_ref = MX::sym("x_ref", state_dim, N + 1); // 参考轨迹

    // 初始化代价函数和约束条件
    MX total_cost = 0;
    std::vector<MX> constraints;

    // 构造代价函数和约束条件
    for (int k = 0; k < N; ++k) {
        // 阶段代价
        std::vector<MX> cost_out = cost({x(Slice(), k), u(Slice(), k), x_ref(Slice(), k)});
        total_cost += cost_out[0];  // 使用索引访问第一个输出

        // 动力学约束
        MXDict dyn_out = dynamics(MXDict{{"x", x(Slice(), k)}, {"u", u(Slice(), k)}}); // 使用 MXDict
        constraints.push_back(x(Slice(), k + 1) - dyn_out.at("x_next"));

        // 控制变量不等式约束
        Function ineq_constraints = build_inequality_constraints(DM::zeros(control_dim, 1), DM::ones(control_dim, 1));
        MXDict ineq_out = ineq_constraints(MXDict{{"u", u(Slice(), k)}});
        constraints.push_back(ineq_out.at("g"));

        // 状态变量不等式约束
        Function state_constraints = build_state_constraints(DM::zeros(state_dim, 1), DM::ones(state_dim, 1));
        MXDict state_out = state_constraints(MXDict{{"x", x(Slice(), k)}});
        constraints.push_back(state_out.at("g"));
    }

    // 终端代价
    std::vector<MX> terminal_cost_out = cost({x(Slice(), N), MX::zeros(control_dim, 1), x_ref(Slice(), N)});
    total_cost += terminal_cost_out[1];  // 索引 1 对应 "l_terminal"

    // 展平约束条件
    MX g_flat = vertcat(constraints);

    // 定义优化变量
    MX opt_variables = vertcat(reshape(x, -1, 1), reshape(u, -1, 1));

    // 定义非线性规划问题，标准的名称，求解器会识别
    MXDict nlp = {
        {"x", opt_variables},// 优化变量
        {"f", total_cost}, // 目标函数
        {"g", g_flat},   // 约束条件
        {"p", vertcat(x0, reshape(x_ref, -1, 1))} // 代表该变量是参数，不参与优化
    };

    // 创建求解器
    Dict opts;
    opts["ipopt.print_level"] = 0; // 禁止 IPOPT 输出
    opts["print_time"] = 0;        // 禁止求解时间输出
    opts["ipopt.tol"] = 1e-6;      // 设置求解精度
    Function solver = nlpsol("solver", "ipopt", nlp, opts);

    // 设置初始值和参考轨迹
    DM x0_val = DM::zeros(state_dim);
    DM x_ref_val = DM::zeros(state_dim, N + 1);
    for (int i = 0; i < N + 1; ++i) {
        x_ref_val(0, i) = i * 0.5;
        x_ref_val(1, i) = 1.0;
    }

    // 初始猜测值
    DM x_guess = DM::repmat(x0_val, 1, N + 1);
    DM u_guess = DM::zeros(control_dim, N);
    DM opt_init = vertcat(reshape(x_guess, -1, 1), reshape(u_guess, -1, 1));
    DM p = vertcat(x0_val, reshape(x_ref_val, -1, 1));

    // 求解问题,提供优化问题的具体数据
    DMDict arg = {
        {"x0", opt_init}, // 初始猜测
        {"p", p},       // 参数
        {"lbg", DM::zeros(g_flat.size1())}, // 约束下界
        {"ubg", DM::zeros(g_flat.size1())}  // 约束上界
    };
    DMDict res = solver(arg);

    // 提取解
    DM opt_sol = res["x"];
    DM x_sol = reshape(opt_sol(Slice(0, state_dim * (N + 1))), state_dim, N + 1);
    DM u_sol = reshape(opt_sol(Slice(state_dim * (N + 1), opt_sol.size1())), control_dim, N);

    // 输出结果
    std::cout << "优化后的状态轨迹:" << std::endl;
    std::cout << x_sol << std::endl;
    std::cout << "优化后的控制输入:" << std::endl;
    std::cout << u_sol << std::endl;
}

// === 修复 iterative_method_demo 中的 MX 检查问题 ===
void iterative_method_demo() {
    // 参数设置
    int N = 20; // 时间步数
    double dt = 0.01; // 时间步长
    int state_dim = 4; // 状态维度
    int control_dim = 2; // 控制维度

    // 状态代价矩阵
    DM Q = DM::diag({10, 10, 1, 1}); // 状态代价权重
    DM R = DM::diag({1, 1});         // 控制代价权重
    DM Qf = DM::diag({10, 10, 1, 1}); // 终端状态代价权重

    // 初始化状态和控制序列
    std::vector<DM> Xc(N + 1, DM::zeros(state_dim)); // 状态序列
    std::vector<DM> Uc(N, DM::zeros(control_dim));   // 控制序列

    // 目标状态
    DM x_goal = DM::zeros(state_dim);
    x_goal(0) = 10.0; // x 方向目标位置
    x_goal(1) = 1.0;  // y 方向目标位置

    // 获取动力学函数（包含维度信息）
    Function dynamics = build_dynamics(dt, state_dim, control_dim);

    // 符号变量（用于 jacobian 计算）
    MX sym_x = MX::sym("x", state_dim);
    MX sym_u = MX::sym("u", control_dim);
    MX sym_next_state = dynamics(std::vector<MX>{sym_x, sym_u})[0];

    // 计算符号雅可比矩阵
    MX df_dx_sym = jacobian(sym_next_state, sym_x);
    MX df_du_sym = jacobian(sym_next_state, sym_u);

    // 将符号函数封装为 CasADi 函数
    Function jacobian_x = Function("jacobian_x", {sym_x, sym_u}, {df_dx_sym});
    Function jacobian_u = Function("jacobian_u", {sym_x, sym_u}, {df_du_sym});

    // 迭代优化
    for (int iter = 0; iter < 10; ++iter) {
        std::cout << "迭代次数: " << iter + 1 << std::endl;
        double total_cost = 0.0; // 初始化总代价

        // 前向积分：计算状态序列
        for (int k = 0; k < N; ++k) {
            auto dyn_out = dynamics(std::vector<DM>{Xc[k], Uc[k]})[0]; // 使用数值化接口
            Xc[k + 1] = dyn_out; // 直接赋值为 DM 类型
        }

        // 后向积分：计算协态变量
        std::vector<DM> lambda(N + 1, DM::zeros(state_dim));
        lambda[N] = mtimes(Qf, Xc[N] - x_goal); // 终端条件
        for (int k = N - 1; k >= 0; --k) {
            DM dL_dx = 2 * mtimes(Q, Xc[k] - x_goal);

            // 使用预定义的符号函数计算雅可比矩阵
            DM df_dx = jacobian_x(std::vector<DM>{Xc[k], Uc[k]})[0];

            lambda[k] = lambda[k + 1] + dt * (dL_dx + mtimes(df_dx.T(), lambda[k + 1]));
        }

        // 更新控制变量
        for (int k = 0; k < N; ++k) {
            DM dL_du = 2 * mtimes(R, Uc[k]);

            // 使用预定义的符号函数计算雅可比矩阵
            DM df_du = jacobian_u(std::vector<DM>{Xc[k], Uc[k]})[0];

            DM grad_H = dL_du + mtimes(df_du.T(), lambda[k + 1]);
            Uc[k] -= 0.01 * grad_H; // 学习率为 0.01
        }

        // 计算当前总代价
        for (int k = 0; k < N; ++k) {
            DM dx = Xc[k] - x_goal; // 状态偏差
            DM stage_cost = mtimes(mtimes(dx.T(), Q), dx) + mtimes(mtimes(Uc[k].T(), R), Uc[k]);
            total_cost += static_cast<double>(stage_cost(0));
        }
        
        DM terminal_cost = mtimes(mtimes((Xc[N] - x_goal).T(), Qf), (Xc[N] - x_goal));
        total_cost += static_cast<double>(terminal_cost(0));

        std::cout << "当前总代价: " << total_cost << std::endl;

        // 收敛判定
        double max_control_update = 0.0;
        for (int k = 0; k < N; ++k) {
            max_control_update = std::max(max_control_update, static_cast<double>(norm_2(Uc[k])(0))); // 转换为标量值
        }
        if (max_control_update < 1e-3) {
            std::cout << "迭代收敛" << std::endl;
            break;
        }
    }

    // 输出最终状态和控制序列
    std::cout << "最终状态序列:" << std::endl;
    for (const auto& x : Xc) {
        std::cout << x << std::endl;
    }
    std::cout << "最终控制序列:" << std::endl;
    for (const auto& u : Uc) {
        std::cout << u << std::endl;
    }
}

// === 使用更全面的定义替代解析式 ===
void cilqr_demo() {
    // 参数设置
    double dt = 0.1; // 时间步长
    int N = 20;      // 时间步数
    int nx = 4;      // 状态维度
    int nu = 2;      // 控制维度

    // 权重矩阵（与 casadi_demo.cpp 的 iLQR 示例一致的量级）
    DM Q = DM::diag(DM(std::vector<double>{100.0, 100.0, 10.0, 1.0}));
    DM R = DM::diag(DM(std::vector<double>{1.0, 0.1}));
    DM Qf = Q * 10.0;

    // 状态和控制变量的约束范围
    DM x_min = DM::vertcat({-10, -10, -M_PI, 0});
    DM x_max = DM::vertcat({10, 10, M_PI, 5});
    DM u_min = DM::vertcat({-1, -0.5});
    DM u_max = DM::vertcat({1, 0.5});

    // 符号变量
    MX x = MX::sym("x", nx);
    MX u = MX::sym("u", nu);

    // 使用已实现的构建函数构造动力学和代价导数函数
    Function dyn_lin = build_dynamics(dt, nx, nu);
    Function cost_der = build_cost(Q, R, nx, nu);

    // 初始化状态和控制序列
    std::vector<DM> X(N + 1, DM::zeros(nx));
    std::vector<DM> U(N, DM::zeros(nu));

    // 设置非零初始值
    X[0] = DM::vertcat({1, 1, 0, 0}); // 初始状态
    for (int i = 0; i < N; ++i) {
        U[i] = DM::vertcat({0.1, 0.1}); // 初始控制
    }

    // iLQR 主循环
    const int max_iter = 100;
    double tol = 1e-3;
    double mu = 1e-6;
    double prev_cost = 1e300;

    for (int iter = 0; iter < max_iter; ++iter) {
        std::cout << "迭代次数: " << iter + 1 << std::endl;

        // 1) 计算导数
        std::vector<DM> Fx_list(N), Fu_list(N);
        std::vector<DM> lx_list(N), lu_list(N), lxx_list(N), luu_list(N), lux_list(N);
        for (int k = 0; k < N; ++k) {
            casadi::DMDict in_dyn, out_dyn;
            in_dyn["x"] = X[k];
            in_dyn["u"] = U[k];
            out_dyn = dyn_lin(in_dyn);
            Fx_list[k] = out_dyn.at("Fx");
            Fu_list[k] = out_dyn.at("Fu");

            casadi::DMDict in_cost, out_cost;
            in_cost["x"] = X[k];
            in_cost["u"] = U[k];
            in_cost["xr"] = DM::zeros(nx, 1);
            out_cost = cost_der(in_cost);
            lx_list[k] = out_cost.at("lx");
            lu_list[k] = out_cost.at("lu");
            lxx_list[k] = out_cost.at("lxx");
            luu_list[k] = out_cost.at("luu");
            lux_list[k] = out_cost.at("lux");
        }

        // 终端代价导数
        DM Vx = 2 * (X[N] - DM::zeros(nx, 1)) * 10.0;
        DM Vxx = DM::diag(DM(std::vector<double>{10.0, 10.0, 10.0, 10.0}));

        // 2) 后向传播
        std::vector<DM> K_list(N), kff_list(N);
        bool diverged = false;
        for (int k = N - 1; k >= 0; --k) {
            DM Fxk = Fx_list[k];
            DM Fuk = Fu_list[k];
            DM lxk = lx_list[k];
            DM luk = lu_list[k];
            DM lxxk = lxx_list[k];
            DM luuk = luu_list[k];
            DM luxk = lux_list[k];

            DM Qx = lxk + mtimes(Fxk.T(), Vx);
            DM Qu = luk + mtimes(Fuk.T(), Vx);
            DM Qxx = lxxk + mtimes(mtimes(Fxk.T(), Vxx), Fxk);
            DM Quu = luuk + mtimes(mtimes(Fuk.T(), Vxx), Fuk);
            DM Qux = luxk + mtimes(mtimes(Fuk.T(), Vxx), Fxk);

            // 正则化
            DM Quu_reg = Quu + mu * DM::eye(nu);
            if (static_cast<double>(det(Quu_reg)) < 1e-6) {
                diverged = true;
                break;
            }

            // DM x = solve(A, b) 等价于 x = A^{-1} * b
            // k=-Quu_reg^{-1} * Qu ->Quu_reg * k = -Qu
            // K=-Quu_reg^{-1} * Qux -> Quu_reg * K = -Qux
            DM kff = -solve(Quu_reg, Qu);
            DM K = -solve(Quu_reg, Qux);

            kff_list[k] = kff;
            K_list[k] = K;

            // Vx=Qx + Qux^T * kff 
            // Vxx=Qxx + Qux.T() * K
            Vx = Qx + mtimes(Qux.T(), kff);
            Vxx = Qxx + mtimes(Qux.T(),K);

            Vxx = 0.5 * (Vxx + Vxx.T()); // 确保对称性
        }

        if (diverged) {
            mu *= 10;
            std::cout << "后向传播发散，增加正则项 mu=" << mu << std::endl;
            continue;
        }

        // 3) 前向传播（线搜索）
        bool accepted = false;
        std::vector<double> alphas = {1.0, 0.5, 0.25, 0.1, 0.05};
        if (iter > 10) {
            alphas = {0.5, 0.25, 0.1, 0.05, 0.01};
        }
        for (double alpha : alphas) {
            std::vector<DM> X_new(N + 1), U_new(N);
            X_new[0] = X[0];
            for (int k = 0; k < N; ++k) {
                DM dx = X_new[k] - X[k];
                U_new[k] = U[k] + alpha * kff_list[k] + mtimes(K_list[k], dx);
                // dyn_lin 输出: x_next, Fx, Fu
                X_new[k + 1] = dyn_lin(std::vector<DM>{X_new[k], U_new[k]}).at(0);
            }

            // 替换为 compute_trajectory_cost 函数
            DM cost = compute_trajectory_cost(X_new, U_new, Q, R, Qf, DM::zeros(nx, 1));

            if (static_cast<double>(cost) < prev_cost) {
                X = X_new;
                U = U_new;
                prev_cost = static_cast<double>(cost);
                accepted = true;
                break;
            }
        }

        if (!accepted) {
            mu *= 10;
            std::cout << "线搜索失败，增加正则项 mu=" << mu << std::endl;
            continue;
        }

        mu = std::max(1e-6, mu / 10);
        if (std::fabs(prev_cost) < tol) {
            std::cout << "收敛，代价=" << prev_cost << std::endl;
            break;
        }
    }

    // 输出结果
    std::cout << "最终状态序列:" << std::endl;
    for (const auto &x : X) {
        std::cout << x << std::endl;
    }
    std::cout << "最终控制序列:" << std::endl;
    for (const auto &u : U) {
        std::cout << u << std::endl;
    }
}

// === 修复 compute_trajectory_cost 中的类型问题 ===
// 将 cost 的类型从 double 改为 casadi::DM
DM compute_trajectory_cost(const std::vector<DM>& X, const std::vector<DM>& U, const DM& Q, const DM& R, const DM& Qf, const DM& x_goal) {
    DM cost = DM(0.0); // 初始化为 casadi::DM 类型
    int N = U.size();
    for (int k = 0; k < N; ++k) {
        DM dx = X[k] - x_goal;
        cost += mtimes(mtimes(dx.T(), Q), dx) + mtimes(mtimes(U[k].T(), R), U[k]);
    }
    DM terminal_cost = mtimes(mtimes((X[N] - x_goal).T(), Qf), (X[N] - x_goal));
    cost += terminal_cost;
    return cost;
}
