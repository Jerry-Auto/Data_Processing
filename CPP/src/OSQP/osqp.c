#include <osqp/osqp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    // 您的数据定义保持不变
    OSQPInt n = 2;
    OSQPInt m = 3;
    
    OSQPFloat P_x[3] = {4.0, 1.0, 2.0};
    OSQPInt P_i[3] = {0, 0, 1};
    OSQPInt P_p[3] = {0, 1, 3};
    OSQPInt P_nnz = 3;

    OSQPFloat A_x[4] = {1.0, 1.0, 1.0, 1.0};
    OSQPInt A_i[4] = {0, 1, 0, 2};
    OSQPInt A_p[3] = {0, 2, 4};
    OSQPInt A_nnz = 4;
    
    // 使用 OSQPCscMatrix_new
    OSQPCscMatrix* P = OSQPCscMatrix_new(n, n, P_nnz, P_x, P_i, P_p);
    OSQPCscMatrix* A = OSQPCscMatrix_new(m, n, A_nnz, A_x, A_i, A_p);

    OSQPFloat q[2] = {1.0, 1.0};
    OSQPFloat l[3] = {1.0, 0.0, 0.0};
    OSQPFloat u[3] = {1.0, 0.7, 0.7};


    // 其余代码保持不变
    OSQPSolver* solver;
    OSQPInt exitflag;
    
    // 设置配置
    OSQPSettings* settings = (OSQPSettings*)malloc(sizeof(OSQPSettings));
    if (settings) {
        osqp_set_default_settings(settings);
    }

    // 设置问题并求解
    exitflag = osqp_setup(&solver, P, q, A, l, u, m, n, settings);
    
    if (exitflag == 0) {
        osqp_solve(solver);
        
        if (solver->info->status_val == OSQP_SOLVED) {
            printf("求解成功！\n");
            printf("目标函数值: %f\n", solver->info->obj_val);
            printf("解: x0 = %f, x1 = %f\n", solver->solution->x[0], solver->solution->x[1]);
        }
        
        // 清理资源
        osqp_cleanup(solver);
    }
    
    // 释放内存
    // 4. 清理资源（使用free）
    // if (A) free(A);
    // if (P) free(P);
    // if (settings) free(settings);

    OSQPCscMatrix_free(P); // 释放P矩阵
    OSQPCscMatrix_free(A); // 释放A矩阵
    OSQPSettings_free(settings); // 释放设置

    return 0;
}