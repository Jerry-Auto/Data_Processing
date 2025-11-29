

#include <iostream>         // 标准输入输出库

#include"AD_demo.h"

// 使用简写别名，提高代码可读性
using CppAD::AD;    // AD作为CppAD::AD的缩写
using std::vector;  // vector作为std::vector的缩写


/**
 * CppAD自动微分库示例程序
 * 功能：计算多项式函数 f(x) = 1 + x + x² + x³ + x⁴ 在 x=3 处的导数
 */
int  scalar_demo(){
    // ==================== 第一步：设置多项式系数 ====================
    size_t k = 5;                   // 多项式系数的个数（对应5次多项式）
    CPPAD_TESTVECTOR(double) a(k);  // 创建多项式系数向量
    
    // 初始化所有多项式系数为1，即构造多项式：f(x) = 1 + x + x² + x³ + x⁴
    for (size_t i = 0; i < k; i++)
      a[i] = 1.;

    // ==================== 第二步：定义自变量并开始记录计算图 ====================
    size_t n = 1;               // 自变量的个数（1元函数）
    vector<AD<double> > ax(n);  // 创建自变量向量（使用自动微分类型）
    ax[0] = 3.;                 // 设置自变量的值为3，在此点计算导数
    
    // 声明ax为自变量，并开始记录计算操作序列（构建计算图）
    CppAD::Independent(ax);

    // ==================== 第三步：计算因变量 ====================
    size_t m = 1;               // 因变量的个数（输出是标量）
    vector<AD<double> > ay(m);  // 创建因变量向量
    
    // 计算多项式函数值，此操作会被记录到计算图中
    ay[0] = Poly(a, ax[0]);

    // ==================== 第四步：停止记录并创建函数对象 ====================
    // 创建ADFun函数对象，封装完整的计算图 f: X → Y
    CppAD::ADFun<double> f(ax, ay);

    // ==================== 第五步：计算导数 ====================
    vector<double> jac(m * n);  // 雅可比矩阵（对于标量函数就是导数，1×1矩阵）
    vector<double> x(n);        // 用于计算导数的输入点
    x[0] = 3.;                  // 设置计算导数的点 x=3
    
    // 计算在x=3处的雅可比矩阵（对于标量函数就是导数f'(3)）
    jac = f.Jacobian(x);
    vector<double> w_f2(1);
    w_f2[0] = 1.0;  // f1的权重

    vector<double> hessian = f.Hessian(x, w_f2);
    // ==================== 第六步：输出结果并验证正确性 ====================
    // 打印自动微分计算的导数结果
    std::cout << "多项式函数 f(x) = 1 + x + x² + x³ + x⁴ 在 x=3 处的导数及二阶导数：" << std::endl;
    std::cout << "f'(3) = " << jac[0] << std::endl;
    std::cout << "f''(3) = " << hessian[0] << std::endl;

    // 验证计算结果是否正确（通过手动计算验证）
    int error_code;  // 返回码：0表示正确，1表示错误
    if (jac[0] == 142.)  // 预期结果：f'(3) = 1 + 2×3 + 3×9 + 4×27 = 142
    {
      std::cout << "✓ 自动微分结果正确！" << std::endl;
      error_code = 0;  // 正确情况的返回码
    }
    else
    {
      std::cout << "✗ 自动微分结果错误！" << std::endl;
      error_code = 1;  // 错误情况的返回码
    }

    // ==================== 附加说明：手动计算验证 ====================
    std::cout << std::endl << "手动验证：" << std::endl;
    std::cout << "原函数: f(x) = 1 + x + x² + x³ + x⁴" << std::endl;
    std::cout << "导函数: f'(x) = 1 + 2x + 3x² + 4x³" << std::endl;
    std::cout << "f'(3) = 1 + 2×3 + 3×9 + 4×27 = " 
              << 1 + 2*3 + 3*9 + 4*27 << std::endl;
    std::cout << "二阶导函数: f'(x) = 2 + 6x + 12x²" << std::endl;
    std::cout << "f'(3) = 2 + 6×3 + 12×9 = " 
              << 2 + 6*3 + 12*9  << std::endl;
    return error_code;  // 返回验证结果
}

int higher_order_demo() {
    using CppAD::AD;
    using CppAD::vector;
    
    // ==================== 第一步：设置多项式系数 ====================
    size_t k = 5;
    CPPAD_TESTVECTOR(double) a(k);
    for (size_t i = 0; i < k; i++)
        a[i] = 1.;

    // ==================== 第二步：定义自变量并开始记录计算图 ====================
    size_t n = 1;
    CPPAD_TESTVECTOR(AD<double>) ax(n);
    ax[0] = 3.;
    CppAD::Independent(ax);

    // ==================== 第三步：计算因变量 ====================
    size_t m = 1;
    CPPAD_TESTVECTOR(AD<double>) ay(m);
    
    // 直接计算多项式，避免未定义的 Poly 函数
    ay[0] = 0.0;
    for(size_t i = 0; i < k; i++) {
        ay[0] += a[i] * CppAD::pow(ax[0], i);
    }

    // ==================== 第四步：停止记录并创建函数对象 ====================
    CppAD::ADFun<double> f(ax, ay);

    // ==================== 第五步：计算各阶导数 ====================
    CPPAD_TESTVECTOR(double) x(n);
    x[0] = 3.;
    size_t order = 5;
    
    // 存储各阶导数
    vector<double> derivatives(order + 1);
    
    std::cout << "多项式函数 f(x) = 1 + x + x² + x³ + x⁴ 在 x=3 处的各阶导数：" << std::endl;
    
    // 计算函数值 (0阶导数)
    CPPAD_TESTVECTOR(double) dy0=f.Forward(0, x);
    derivatives[0] = dy0[0];  // 使用 Value() 而不是 Value[]
    std::cout << "f(3) = " << derivatives[0] << " (函数值)" << std::endl;
    
    // 计算高阶导数
    for(size_t o = 1; o <= order; o++) {
        CPPAD_TESTVECTOR(double) dx(n);
        for(size_t i = 0; i < n; i++) {
            dx[i] = 0.0;
        }
        if(o == 1) dx[0] = 1.0;
        
        CPPAD_TESTVECTOR(double) dy = f.Forward(o, dx);
        double taylor_coeff = dy[0];
        
        // 转换为实际导数：f^(k)(x) = k! * 泰勒系数
        double derivative = taylor_coeff;
        for(size_t i = 2; i <= o; i++) {
            derivative *= i;
        }
        derivatives[o] = derivative;
        
        std::cout << "f^(" << o << ")(3) = " << derivative;
        
        // 验证正确性
        if(o <= 4) {
            double expected = 0.0;
            for(size_t i = o; i < k; i++) {
                double term = a[i];
                for(size_t j = 0; j < o; j++) {
                    term *= (i - j);
                }
                term *= std::pow(x[0], i - o);
                expected += term;
            }
            std::cout << " (期望: " << expected << ")";
            if(std::abs(derivative - expected) < 1e-10) {
                std::cout << " ✓";
            } else {
                std::cout << " ✗";
            }
        } else {
            if(std::abs(derivative) < 1e-10) {
                std::cout << " ✓ (正确：应为0)";
            } else {
                std::cout << " ✗ (错误：应为0)";
            }
        }
        std::cout << std::endl;
    }
    
    return 0;
}

int vector_demo() {
  std::cout << "=== 多变量函数导数计算 ===" << std::endl;
  
  // ==================== 第一部分：在点(1,2)计算 ====================
  {
    // 步骤1-2：定义自变量并开始记录
    vector<AD<double>> ax(2);
    ax[0] = 1.0; ax[1] = 2.0;
    CppAD::Independent(ax);

    // 步骤3：计算函数值
    vector<AD<double>> ay(2);
    ay[0] = ax[0]*ax[0] + ax[1]*ax[1];  // f1(x,y) = x² + y²
    ay[1] = ax[0]*ax[1];                // f2(x,y) = xy

    // 步骤4：创建函数对象
    CppAD::ADFun<double> f(ax, ay);

    // 步骤5：计算雅可比矩阵
    vector<double> x(2);
    x[0] = 1.0; x[1] = 2.0;
    vector<double> jac = f.Jacobian(x);

    // 输出梯度结果
    std::cout << "计算在点：(" << x[0] << "," << x[1] << ")处的梯度：" << std::endl;
    std::cout << "梯度: [∂f1/∂x, ∂f1/∂y] = [" << jac[0] << ", " << jac[1] << "]" << std::endl;
    std::cout << "梯度: [∂f2/∂x, ∂f2/∂y] = [" << jac[2] << ", " << jac[3] << "]" << std::endl;

    // ==================== 计算Hessian矩阵 ====================
    std::cout << "\n=== 计算Hessian矩阵 ===" << std::endl;
    
    // 方法1：使用权重向量选择输出
    std::cout << "\n方法1：使用原多输出函数分别计算Hessian" << std::endl;
    
    vector<double> w_f1(2);
    w_f1[0] = 1.0;  // f1的权重
    w_f1[1] = 0.0;  // f2的权重（忽略）
    vector<double> hessian_f1 = f.Hessian(x, w_f1);
    
    std::cout << "f1的Hessian矩阵:" << std::endl;
    std::cout << "[" << hessian_f1[0] << ", " << hessian_f1[1] << "]" << std::endl;
    std::cout << "[" << hessian_f1[2] << ", " << hessian_f1[3] << "]" << std::endl;
    
    vector<double> w_f2(2);
    w_f2[0] = 0.0;  // f1的权重（忽略）
    w_f2[1] = 1.0;  // f2的权重
    vector<double> hessian_f2 = f.Hessian(x, w_f2);
    
    std::cout << "f2的Hessian矩阵:" << std::endl;
    std::cout << "[" << hessian_f2[0] << ", " << hessian_f2[1] << "]" << std::endl;
    std::cout << "[" << hessian_f2[2] << ", " << hessian_f2[3] << "]" << std::endl;

  // ==================== 第二部分：在点(2,1)计算 ====================
    // 在新点计算
    vector<double> x_new(2);
    x_new[0] = 2.0; x_new[1] = 1.0;
    vector<double> jac_new = f.Jacobian(x_new);
    
    std::cout << "\n计算在点：(" << x_new[0] << "," << x_new[1] << ")处的梯度：" << std::endl;
    std::cout << "梯度: [∂f1/∂x, ∂f1/∂y] = [" << jac_new[0] << ", " << jac_new[1] << "]" << std::endl;
    std::cout << "梯度: [∂f2/∂x, ∂f2/∂y] = [" << jac_new[2] << ", " << jac_new[3] << "]" << std::endl;

    vector<double> hessian_f1_new = f.Hessian(x_new, w_f1);
    vector<double> hessian_f2_new = f.Hessian(x_new, w_f2);
    
    std::cout << "f1的Hessian矩阵:" << std::endl;
    std::cout << "[" << hessian_f1_new[0] << ", " << hessian_f1_new[1] << "]" << std::endl;
    std::cout << "[" << hessian_f1_new[2] << ", " << hessian_f1_new[3] << "]" << std::endl;
    
    std::cout << "f2的Hessian矩阵:" << std::endl;
    std::cout << "[" << hessian_f2_new[0] << ", " << hessian_f2_new[1] << "]" << std::endl;
    std::cout << "[" << hessian_f2_new[2] << ", " << hessian_f2_new[3] << "]" << std::endl;
  }

  {
    // 方法2：分别构建单输出函数（推荐，避免权重向量复杂性）
    std::cout << "方法2：分别计算每个输出的Hessian" << std::endl;

    vector<double> x(2);
    x[0] = 1.0; x[1] = 2.0;
    // 步骤1-2：定义自变量并开始记录
    vector<AD<double>> ax(2);
    ax[0] = 1.0; ax[1] = 2.0;

    CppAD::Independent(ax);

    // 对于f1(x,y) = x² + y²
    vector<AD<double>> ay1(1);
    ay1[0] = ax[0]*ax[0] + ax[1]*ax[1];
    CppAD::ADFun<double> f1(ax, ay1);
    
    
    // 计算f1的Hessian
    vector<double> w1(1, 1.0);  // 权重向量
    vector<double> hessian1 = f1.Hessian(x, w1);
    
    std::cout << "f1的Hessian矩阵 (在点(" << x[0] << "," << x[1] << ")):" << std::endl;
    std::cout << "[" << hessian1[0] << ", " << hessian1[1] << "]" << std::endl;
    std::cout << "[" << hessian1[2] << ", " << hessian1[3] << "]" << std::endl;
    
    // 对于f2(x,y) = xy 
    vector<AD<double>> ax1(2);
    ax1[0] = 1.0; ax1[1] = 2.0; 

    CppAD::Independent(ax1);

    vector<AD<double>> ay2(1);
    ay2[0] = ax1[0]*ax1[1];
    CppAD::ADFun<double> f2(ax1, ay2);
    // 计算f2的Hessian
    vector<double> w2(1, 1.0);  // 权重向量
    vector<double> hessian2 = f2.Hessian(x, w2);
    
    std::cout << "f2的Hessian矩阵 (在点(" << x[0] << "," << x[1] << ")):" << std::endl;
    std::cout << "[" << hessian2[0] << ", " << hessian2[1] << "]" << std::endl;
    std::cout << "[" << hessian2[2] << ", " << hessian2[3] << "]" << std::endl;

  }
  return 0;
}

int auto_sparse_demo() {
    std::cout << "=== 自动稀疏模式检测示例 ===" << std::endl;
    
    // 定义一个大尺寸的稀疏函数
    size_t n = 100;  // 100个自变量
    size_t m = 50;   // 50个因变量
    
    vector<AD<double>> ax(n);
    for(size_t i = 0; i < n; i++) {
        ax[i] = double(i + 1);
    }
    CppAD::Independent(ax);
    
    vector<AD<double>> ay(m);
    // 创建一个稀疏函数：每个输出只依赖于少数几个输入
    for(size_t i = 0; i < m; i++) {
        ay[i] = 0.0;
        // 每个f_i只依赖于x_i, x_{i+1}, x_{i+2}
        for(size_t j = i; j < std::min(i + 3, n); j++) {
            ay[i] += ax[j] * ax[j];
        }
    }
    
    CppAD::ADFun<double> f(ax, ay);
    f.optimize();  // 优化计算图
    

    // 精确的稀疏模式检测
    // 自动检测稀疏模式
    // 方法1：使用 ForSparseJac（推荐）
    std::cout << "方法1：使用 ForSparseJac 检测稀疏模式" << std::endl;
    vector<bool> select_domain(n, true);  // 选择所有自变量
    vector<bool> pattern_forward = f.ForSparseJac(m, select_domain);
    
    // 方法2：使用 RevSparseJac（需要正确设置参数）
    std::cout << "方法2：使用 RevSparseJac 检测稀疏模式" << std::endl;
    vector<bool> select_range(m * n, true);  // ⭐️ 修复：大小应为 m × n ⭐️
    vector<bool> pattern_reverse = f.RevSparseJac(n, select_range);
    
    // 使用其中一种模式继续计算
    vector<bool> pattern = pattern_forward;  // 选择前向模式的结果
 
    std::cout << "雅可比矩阵维度: " << m << " × " << n << " = " << m * n << " 个元素" << std::endl;
    
    // 统计非零元素数量
    size_t non_zeros = 0;
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            if(pattern[i * n + j]) {
                non_zeros++;
            }
        }
    }
    
    std::cout << "非零元素数量: " << non_zeros << std::endl;
    std::cout << "稀疏度: " << (100.0 * non_zeros) / (m * n) << "%" << std::endl;
    
    // 使用稀疏模式计算雅可比
    vector<double> x(n);
    for(size_t i = 0; i < n; i++) x[i] = double(i + 1);
    
    CppAD::sparse_jacobian_work work;
    vector<size_t> row, col;
    vector<double> val;
    
    // 对于输入少输出多的问题，使用反向模式
    if (n < m) {
        f.SparseJacobianReverse(x, pattern, row, col, val, work);
    } 
    // 对于输入多输出少的问题，使用前向模式  
    else {
        f.SparseJacobianForward(x, pattern, row, col, val, work);
    }

    std::cout << "实际计算的非零元素: " << val.size() << std::endl;
    
    // 显示前10个非零元素
    std::cout << "前10个非零元素:" << std::endl;
    for(size_t k = 0; k < std::min(size_t(10), val.size()); k++) {
        std::cout << "J[" << row[k] << "][" << col[k] << "] = " << val[k] << std::endl;
    }
    return 0;
}


bool simple_sparse_jacobian(void)
{  
    bool ok = true;
    using CppAD::AD;
    typedef CPPAD_TESTVECTOR(size_t)     SizeVector;
    typedef CppAD::sparse_rc<SizeVector> sparsity;
    
    // 构建稀疏函数
    size_t n = 100, m = 50;
    CPPAD_TESTVECTOR(AD<double>) ax(n);
    for(size_t i = 0; i < n; i++) ax[i] = double(i + 1);
    CppAD::Independent(ax);
    
    CPPAD_TESTVECTOR(AD<double>) ay(m);
    for(size_t i = 0; i < m; i++) {
        ay[i] = 0.0;
        for(size_t j = i; j < std::min(i + 3, n); j++) {
            ay[i] += ax[j] * ax[j];
        }
    }
    
    CppAD::ADFun<double> f(ax, ay);
    f.optimize();
    
    // 使用 for_jac_sparsity 检测稀疏模式
    sparsity pattern_in(n, n, n);  // n×n 单位矩阵
    for(size_t k = 0; k < n; k++) pattern_in.set(k, k, k);
    
    sparsity pattern_out;
    f.for_jac_sparsity(pattern_in, false, false, false, pattern_out);
    
    std::cout << "雅可比矩阵: " << m << " × " << n << std::endl;
    std::cout << "非零元素: " << pattern_out.nnz() << std::endl;
    
    // 计算稀疏雅可比
    CPPAD_TESTVECTOR(double) x(n);
    for(size_t i = 0; i < n; i++) x[i] = double(i + 1);
    
    // 创建稀疏矩阵容器
    CppAD::sparse_rcv<SizeVector, CPPAD_TESTVECTOR(double)> subset(pattern_out);
    
    // 创建工作结构
    CppAD::sparse_jac_work work;
    
    // 调用 sparse_jac_rev - 修正后的调用方式
    f.sparse_jac_rev(x, subset, pattern_out, "cppad", work);
    
    std::cout << "计算完成，验证 " << std::min(size_t(3), subset.nnz()) << " 个元素:" << std::endl;
    
    // 验证结果
    const CPPAD_TESTVECTOR(double)& values = subset.val();
    const SizeVector& rows = subset.row();
    const SizeVector& cols = subset.col();
    
    for(size_t k = 0; k < std::min(size_t(3), subset.nnz()); k++) {
        double computed = values[k];
        double expected = 2.0 * x[cols[k]];
        ok &= CppAD::NearEqual(computed, expected, 1e-10, 1e-10);
        std::cout << "J[" << rows[k] << "][" << cols[k] << "] = " 
                  << computed << " (期望: " << expected << ")" << std::endl;
    }
    
    return ok;
}



int sparse_jac_demo() {
    using CppAD::AD;
    using CppAD::vector;
    
    // 定义函数维度
    size_t n = 4; // 输入变量个数
    size_t m = 3; // 输出变量个数
    
    std::cout << "函数: R^" << n << " -> R^" << m << std::endl;
    
    // 1. 定义自变量并开始记录
    vector<AD<double>> X(n);
    for(size_t i = 0; i < n; i++) {
        X[i] = 1.0 + 0.1 * i; // 初始值
    }
    CppAD::Independent(X);
    
    // 2. 定义因变量 (构建一个稀疏函数)
    vector<AD<double>> Y(m);
    
    // 故意创建稀疏依赖关系：
    // Y0 = X0 * X1        (只依赖 X0, X1)
    // Y1 = X1 * X2        (只依赖 X1, X2)  
    // Y2 = X2 * X3        (只依赖 X2, X3)
    Y[0] = X[0] * X[1];
    Y[1] = X[1] * X[2];
    Y[2] = X[2] * X[3];
    
    // 3. 创建 ADFun 对象
    CppAD::ADFun<double> f(X, Y);
    
    // 4. 计算点在 x = (1.0, 1.1, 1.2, 1.3)
    vector<double> x(n);
    for(size_t i = 0; i < n; i++) {
        x[i] = 1.0 + 0.1 * i;
    }
    
    // 5. 使用 ForSparseJac 计算稀疏模式
    // 创建单位矩阵作为输入模式
    vector<bool> r(n * n);
    for(size_t i = 0; i < n; i++) {
        for(size_t j = 0; j < n; j++) {
            r[i * n + j] = (i == j); // 单位矩阵
        }
    }
    
    // 计算雅可比矩阵的稀疏模式
    vector<bool> pattern = f.ForSparseJac(n, r);
    
    // 6. 打印稀疏模式
    std::cout << "\n雅可比矩阵稀疏模式 (0=零, 1=非零):" << std::endl;
    std::cout << "   ";
    for(size_t j = 0; j < n; j++) {
        std::cout << "X" << j << " ";
    }
    std::cout << std::endl;
    
    for(size_t i = 0; i < m; i++) {
        std::cout << "Y" << i << " ";
        for(size_t j = 0; j < n; j++) {
            std::cout << " " << pattern[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
    
    // 7. 准备 SparseJacobianForward 所需的参数
    // 提取非零元素的位置
    vector<size_t> row, col;
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            if(pattern[i * n + j]) {
                row.push_back(i);
                col.push_back(j);
            }
        }
    }
    
    size_t K = row.size(); // 非零元素个数
    
    // 8. 使用 SparseJacobianForward 计算具体数值
    vector<double> jac(K);    // 存储雅可比矩阵的非零值
    CppAD::sparse_jacobian_work work; // 工作结构，用于提高效率
    CppAD::sparse_jacobian_work work1; // 工作结构，用于提高效率

    vector<double> jac3(K);    // 存储雅可比矩阵的非零值
    
    // 计算稀疏雅可比矩阵
    size_t n_sweep = f.SparseJacobianForward(x, pattern, row, col, jac, work);
    size_t n_sweep1 = f.SparseJacobianReverse(x, pattern, row, col, jac3, work1);

    // 方法1: 自动检测模式并计算
    vector<double> jac1 = f.SparseJacobian(x);
    vector<double> jac2 = f.SparseJacobian(x, pattern);

    // 9. 输出结果
    std::cout << "\n稀疏雅可比矩阵数值:" << std::endl;
    std::cout << "非零元素个数: " << K << std::endl;
    std::cout << "SparseJacobianForward计算使用的扫描次数: " << n_sweep << std::endl;
    std::cout << "SparseJacobianReverse计算使用的扫描次数: " << n_sweep1 << std::endl;
    std::cout<<"使用复杂方法SparseJacobianForward：\n";
    for(size_t k = 0; k < K; k++) {
        std::cout << "J[" << row[k] << "][" << col[k] << "] = " 
                  << jac[k] << std::endl;
    }
    std::cout<<"使用复杂方法SparseJacobianReverse：\n";
    for(size_t k = 0; k < K; k++) {
        std::cout << "J[" << row[k] << "][" << col[k] << "] = " 
                  << jac3[k] << std::endl;
    }
    // 修复：正确使用 SparseJacobian 的结果
    std::cout << "\nSparseJacobian (自动模式) - 从稠密矩阵提取:" << std::endl;
    for(size_t k = 0; k < K; k++) {
        size_t i = row[k], j = col[k];
        std::cout << "  J[" << i << "][" << j << "] = " 
                  << jac1[i * n + j] << std::endl;
    }
    
    std::cout << "\nSparseJacobian (提供模式) - 从稠密矩阵提取:" << std::endl;
    for(size_t k = 0; k < K; k++) {
        size_t i = row[k], j = col[k];
        std::cout << "  J[" << i << "][" << j << "] = " 
                  << jac2[i * n + j] << std::endl;
    }

    // 10. 验证结果正确性
    std::cout << "\n验证 (解析解):" << std::endl;
    std::cout << "J[0][0] = ∂Y0/∂X0 = X1 = " << x[1] << std::endl;
    std::cout << "J[0][1] = ∂Y0/∂X1 = X0 = " << x[0] << std::endl;
    std::cout << "J[1][1] = ∂Y1/∂X1 = X2 = " << x[2] << std::endl;
    std::cout << "J[1][2] = ∂Y1/∂X2 = X1 = " << x[1] << std::endl;
    std::cout << "J[2][2] = ∂Y2/∂X2 = X3 = " << x[3] << std::endl;
    std::cout << "J[2][3] = ∂Y2/∂X3 = X2 = " << x[2] << std::endl;
    
    return 0;
}


int multi_output_sparse_hessian() {
    std::cout << "=== 多输出函数稀疏Hessian计算 ===" << std::endl;
    
    const size_t num_inputs = 3;
    const size_t num_outputs = 2;
    
    vector<AD<double>> inputs(num_inputs);
    inputs[0] = 1.0; inputs[1] = 2.0; inputs[2] = 3.0;
    CppAD::Independent(inputs);
    
    vector<AD<double>> outputs(num_outputs);
    // 输出1: f1(x,y,z) = x² + y²
    outputs[0] = inputs[0]*inputs[0] + inputs[1]*inputs[1];
    // 输出2: f2(x,y,z) = x*z + y*z
    outputs[1] = inputs[0]*inputs[2] + inputs[1]*inputs[2];
    
    CppAD::ADFun<double> func(inputs, outputs);
    func.optimize();
    
    vector<double> eval_point(num_inputs);
    eval_point[0] = 1.0; eval_point[1] = 2.0; eval_point[2] = 3.0;
    
    // 分别计算每个输出的Hessian
    for(size_t output_idx = 0; output_idx < num_outputs; output_idx++) {
        std::cout << "\n--- 输出 f" << output_idx << " 的Hessian ---" << std::endl;
        
        vector<bool> select_domain(num_inputs, true);
        vector<bool> select_range(num_outputs, false);
        select_range[output_idx] = true;  // 选择当前输出
        
        vector<bool> sparsity = func.ForSparseHes(select_domain, select_range);
        
        std::cout << "稀疏模式:" << std::endl;
        for(size_t i = 0; i < num_inputs; i++) {
            for(size_t j = 0; j < num_inputs; j++) {
                std::cout << (sparsity[i * num_inputs + j] ? "1 " : "0 ");
            }
            std::cout << std::endl;
        }
        
        // 计算稀疏Hessian
        vector<double> weights(num_outputs, 0.0);
        weights[output_idx] = 1.0;  // 只选择当前输出
        
        CppAD::sparse_hessian_work work;
        vector<size_t> row, col;
        vector<double> val;
        
        func.SparseHessian(eval_point, weights, sparsity, row, col, val, work);
        
        std::cout << "非零元素:" << std::endl;
        for(size_t k = 0; k < val.size(); k++) {
            std::cout << "H[" << row[k] << "][" << col[k] << "] = " << val[k] << std::endl;
        }
    }
    
    return 0;
}