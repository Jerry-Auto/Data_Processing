#include <iostream>
#include"eigen_learn.h"
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;
 
int simple_demo()
{
    cout << "=== Demo 0: 综合示例===" << endl;
    // 1. 矩阵的定义
    // MatrixXd: 动态大小的双精度矩阵，尺寸在运行时确定
    Eigen::MatrixXd m(2, 2);
    // Vector3d: 固定大小的3维双精度列向量
    Eigen::Vector3d vec3d;
    // Vector4d: 固定大小的4维双精度列向量，直接初始化
    Eigen::Vector4d vec4d(1.0, 2.0, 3.0, 4.0);
 
    // 2. 动态矩阵和静态矩阵
    Eigen::MatrixXd matrixXd;  // 未指定尺寸的动态矩阵，初始为0x0
    Eigen::Matrix3d matrix3d;  // 固定大小的3x3双精度矩阵
 
    // 3. 矩阵元素的访问
    // 使用括号运算符()访问和修改矩阵元素（索引从0开始）
    m(0, 0) = 1;      // 第0行第0列赋值为1
    m(0, 1) = 2;      // 第0行第1列赋值为2
    m(1, 0) = m(0, 0) + 3;   // 第1行第0列 = 1 + 3 = 4
    m(1, 1) = m(0, 0) * m(0, 1);  // 第1行第1列 = 1 × 2 = 2
    std::cout << "通过元素访问赋值的矩阵m:" << std::endl << m << std::endl << std::endl;
 
    // 4. 设置矩阵的元素 - 逗号初始化器
    // 使用移位运算符<<按行优先顺序初始化矩阵
    m << -1.5, 2.4,
         6.7, 2.0;
    std::cout << "逗号初始化后的矩阵m:" << std::endl << m << std::endl << std::endl;
    
    // 动态矩阵的初始化
    int row = 4;
    int col = 5;
    Eigen::MatrixXf matrixXf(row, col);  // 4x5单精度动态矩阵
    // 使用逗号初始化器为动态矩阵赋值
    matrixXf << 1, 2, 3, 4, 5, 
                6, 7, 8, 9, 10, 
                11, 12, 13, 14, 15, 
                16, 17, 18, 19, 20;
    std::cout << "4x5动态矩阵matrixXf:" << std::endl << matrixXf << std::endl << std::endl;
    
    // 将矩阵设置为单位矩阵（非方阵时主对角线为1，其余为0）
    matrixXf = Eigen::MatrixXf::Identity(row, col);
    std::cout << "单位矩阵matrixXf:" << std::endl << matrixXf << std::endl << std::endl;
 
    // 5. 重置矩阵大小
    Eigen::MatrixXd matrixXd1(3, 3);  // 创建3x3矩阵
    m = matrixXd1;  // 动态矩阵可以调整大小，这里m从2x2变为3x3
    std::cout << "调整大小后m的维度: " << m.rows() << " x " << m.cols() << std::endl << std::endl;
 
    // 6. 矩阵运算
    // 重新初始化矩阵m为3x3
    m << 1, 2, 7,
         3, 4, 8,
         5, 6, 9;
    
    std::cout << "重新初始化的矩阵m:" << std::endl << m << std::endl;
    
    // 矩阵与随机矩阵相加
    matrixXd1 = Eigen::Matrix3d::Random();  // 生成3x3随机矩阵
    m += matrixXd1;  // 矩阵加法（逐元素相加）
    std::cout << "m + 随机矩阵:" << std::endl << m << std::endl << std::endl;
    
    m *= 2;  // 矩阵标量乘法（每个元素乘以2）
    std::cout << "m * 2:" << std::endl << m << std::endl << std::endl;
    
    // 一元运算符
    std::cout << "-m (取负):" << std::endl << -m << std::endl << std::endl;
    std::cout << "原始矩阵m保持不变:" << std::endl << m << std::endl << std::endl;
 
    // 7. 矩阵的转置、共轭矩阵、伴随矩阵
    std::cout << "m的转置矩阵:" << std::endl << m.transpose() << std::endl << std::endl;
    std::cout << "m的共轭矩阵（实数矩阵与转置相同）:" << std::endl << m.conjugate() << std::endl << std::endl;
    std::cout << "m的伴随矩阵（实数矩阵等价于转置）:" << std::endl << m.adjoint() << std::endl << std::endl;
    std::cout << "原始矩阵m保持不变:" << std::endl << m << std::endl << std::endl;
    
    // 原地转置（直接修改原矩阵，仅适用于方阵）
    m.transposeInPlace();
    std::cout << "原地转置后的m:" << std::endl << m << std::endl << std::endl;
 
    // 8. 矩阵相乘、矩阵向量相乘
    std::cout << "m * m (矩阵乘法):" << std::endl << m * m << std::endl << std::endl;
    
    vec3d = Eigen::Vector3d(1, 2, 3);  // 初始化3维向量
    std::cout << "m * vec3d (矩阵向量乘法):" << std::endl << m * vec3d << std::endl << std::endl;
    
    // 行向量乘以矩阵：需要先将列向量转置为行向量
    std::cout << "vec3d^T * m (行向量乘矩阵):" << std::endl << vec3d.transpose() * m << std::endl << std::endl;
 
    // 9. 矩阵的块操作
    // 恢复m的原始值用于演示块操作
    m << 1, 2, 7,
         3, 4, 8,
         5, 6, 9;
         
    std::cout << "用于块操作的矩阵m:" << std::endl << m << std::endl << std::endl;
    
    // 提取2x2子块（从第1行第1列开始）
    std::cout << "m.block(1,1,2,2) 2x2子块:" << std::endl << m.block(1, 1, 2, 2) << std::endl << std::endl;
    
    // 提取固定大小的1x2子块（从第0行第0列开始）
    std::cout << "m.block<1,2>(0,0) 1x2子块:" << std::endl << m.block<1, 2>(0, 0) << std::endl << std::endl;
    
    // 提取整列
    std::cout << "m的第1列:" << std::endl << m.col(1) << std::endl << std::endl;
    
    // 提取整行
    std::cout << "m的第0行:" << std::endl << m.row(0) << std::endl << std::endl;
 
    // 10. 向量的块操作
    Eigen::ArrayXf arrayXf(10);  // 创建10维动态数组
    arrayXf << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;  // 初始化数组
    
    std::cout << "向量vec3d:" << std::endl << vec3d << std::endl << std::endl;
    std::cout << "数组arrayXf:" << std::endl << arrayXf << std::endl << std::endl;
    
    // 向量块操作
    std::cout << "arrayXf的前5个元素:" << std::endl << arrayXf.head(5) << std::endl << std::endl;
    std::cout << "arrayXf的后4个元素乘以2:" << std::endl << arrayXf.tail(4) * 2 << std::endl << std::endl;
 
    // 11. 求解矩阵的特征值和特征向量
    Eigen::Matrix2f matrix2f;
    matrix2f << 1, 2, 
                3, 4;  // 2x2测试矩阵
    
    // 创建特征值求解器（适用于自伴随矩阵/实对称矩阵）
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2f> eigenSolver(matrix2f);
    
    // 检查求解是否成功
    if (eigenSolver.info() == Eigen::Success) {
        std::cout << "矩阵的特征值:" << std::endl << eigenSolver.eigenvalues() << std::endl << std::endl;
        std::cout << "对应的特征向量（每列一个特征向量）:" << std::endl << eigenSolver.eigenvectors() << std::endl << std::endl;
    } else {
        std::cout << "特征值求解失败!" << std::endl;
    }
 
    // 12. 类Map及动态矩阵的使用
    int array1[4] = {1, 2, 3, 4};    // 第一个矩阵的数据（2x2）
    int array2[4] = {5, 6, 7, 8};    // 第二个矩阵的数据（2x2）
    int array3[4] = {0, 0, 0, 0};    // 结果矩阵的存储空间
    
    // 使用模板函数进行矩阵乘法
    matrix_mul_matrix(array1, 2, 2, array2, 2, 2, array3);
    
    std::cout << "矩阵乘法结果: ";
    for (int i = 0; i < 4; i++)
        std::cout << array3[i] << " ";
    std::cout << std::endl;
    
    // 验证结果：1 * 5+2 * 7=19, 1 * 6+2 * 8=22, 3 * 5+4 * 7=43, 3 * 6+4 * 8=50
    // 因此结果应为: 19, 22, 43, 50
 
    return 0;
}


int init_matrix() {
    cout << "=== Demo 1: 矩阵定义与初始化===" << endl;
    
    // 1. 动态矩阵的定义
    MatrixXd dynamic_mat(2, 2);  // 2x2动态双精度矩阵
    Vector3d vec3d;              // 3维双精度列向量
    Vector4d vec4d(1.0, 2.0, 3.0, 4.0);  // 4维向量直接初始化
    
    // 2. 静态矩阵的定义
    Matrix3d static_mat;         // 3x3静态双精度矩阵
    
    // 3. 矩阵元素的访问和赋值
    dynamic_mat(0, 0) = 1;
    dynamic_mat(0, 1) = 2;
    dynamic_mat(1, 0) = dynamic_mat(0, 0) + 3; 
    dynamic_mat(1, 1) = dynamic_mat(0, 0) * dynamic_mat(0, 1);
    cout << "通过元素访问赋值的矩阵:\n" << dynamic_mat << endl;
    
    // 4. 逗号初始化器
    dynamic_mat << -1.5, 2.4,
                   6.7, 2.0;
    cout << "逗号初始化后的矩阵:\n" << dynamic_mat << endl;
    
    // 5. 大矩阵的初始化
    int row = 4, col = 5;
    MatrixXf large_matrix(row, col);
    large_matrix << 1, 2, 3, 4, 5, 
                    6, 7, 8, 9, 10, 
                    11, 12, 13, 14, 15, 
                    16, 17, 18, 19, 20;
    cout << "4x5大矩阵:\n" << large_matrix << endl;
    
    // 6. 特殊矩阵初始化
    large_matrix = MatrixXf::Identity(row, col);  // 单位矩阵
    cout << "单位矩阵:\n" << large_matrix << endl;
    
    // 7. 矩阵大小调整
    MatrixXd resize_demo(3, 3);
    dynamic_mat = resize_demo;  // 动态矩阵可以调整大小
    cout << "调整大小后的矩阵维度: " << dynamic_mat.rows() 
         << " x " << dynamic_mat.cols() << endl;
    
    return 0;
}



int matrix_operate() {
    cout << "=== Demo 2: 矩阵运算与代数操作===" << endl;
    
    // 创建测试矩阵
    Matrix3d A, B;
    A << 1, 2, 7,
         3, 4, 8,
         5, 6, 9;
    
    B = Matrix3d::Random();  // 随机矩阵
    
    cout << "矩阵 A:\n" << A << endl;
    cout << "随机矩阵 B:\n" << B << endl;
    
    // 1. 基础算术运算
    Matrix3d C = A + B;
    cout << "A + B:\n" << C << endl;
    
    A += B;  // 复合赋值运算
    cout << "A += B 后的 A:\n" << A << endl;
    
    A *= 2;  // 标量乘法
    cout << "A *= 2 后的 A:\n" << A << endl;
    
    // 2. 一元操作符
    cout << "-A (取负):\n" << -A << endl;
    
    // 3. 线性代数运算
    cout << "A的转置:\n" << A.transpose() << endl;
    cout << "A的共轭矩阵:\n" << A.conjugate() << endl;  // 实数矩阵共轭为本身
    cout << "A的伴随矩阵:\n" << A.adjoint() << endl;    // 实数矩阵伴随等于转置
    
    // 4. 原地转置操作
    Matrix3d original_A = A;
    A.transposeInPlace();  // 原地转置，避免复制
    cout << "原地转置后的 A:\n" << A << endl;
    
    // 5. 矩阵乘法
    A = original_A;  // 恢复原始矩阵
    Matrix3d product = A * A;
    cout << "A * A:\n" << product << endl;
    
    // 6. 矩阵向量乘法
    Vector3d v(1, 2, 3);
    Vector3d result = A * v;
    cout << "A * v:\n" << result << endl;
    
    // 7. 向量转置乘法
    RowVector3d row_result = v.transpose() * A;
    cout << "v^T * A:\n" << row_result << endl;
    
    return 0;
}


int block_col_row() {
    cout << "=== Demo 3: 块操作与切片===" << endl;
    
    // 创建测试矩阵
    MatrixXd mat(3, 3);
    mat << 1, 2, 7,
           3, 4, 8,
           5, 6, 9;
    
    cout << "原始矩阵 mat:\n" << mat << endl;
    
    // 1. 基本块操作
    cout << "从(1,1)开始的2x2块:\n" << mat.block(1, 1, 2, 2) << endl;
    cout << "从(0,0)开始的1x2块(固定大小):\n" << mat.block<1, 2>(0, 0) << endl;
    
    // 2. 行和列操作
    cout << "第1列:\n" << mat.col(1) << endl;
    cout << "第0行:\n" << mat.row(0) << endl;
    
    // 3. 向量块操作
    ArrayXf arr(10);
    arr << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
    cout << "数组 arr:\n" << arr.transpose() << endl;
    cout << "前5个元素: " << arr.head(5).transpose() << endl;
    cout << "后4个元素乘以2: " << (arr.tail(4) * 2).transpose() << endl;
    
    // 4. 块操作赋值
    mat.block(1, 1, 2, 2) = Matrix2d::Identity();  // 将中间2x2块设为单位矩阵
    cout << "修改块后的矩阵:\n" << mat << endl;
    
    // 5. 分段操作
    arr.segment(1, 4) *= 2;  // 将索引1开始的4个元素乘以2
    cout << "分段操作后的数组: " << arr.transpose() << endl;
    
    return 0;
}

#include <Eigen/Eigenvalues>



int eigenvalue_map() {
    cout << "=== Demo 4: 特征值分解与Map类 ===" << endl;
    
    // 1. 特征值分解
    Matrix2f mat;
    mat << 1, 2, 
           3, 4;
    
    cout << "测试矩阵:\n" << mat << endl;
    
    SelfAdjointEigenSolver<Matrix2f> eigensolver(mat);
    if (eigensolver.info() == Success) {
        cout << "特征值:\n" << eigensolver.eigenvalues() << endl;
        cout << "特征向量:\n" << eigensolver.eigenvectors() << endl;
    }
    
    // 2. Map类的使用 - 原生数组与Eigen矩阵的零拷贝交互
    cout << "\n=== Map类示例 ===" << endl;
    
    // 定义原生数组
    int array1[4] = {1, 2, 3, 4};
    int array2[4] = {5, 6, 7, 8};
    int array3[4] = {0, 0, 0, 0};
    
    // 使用Map将数组映射为Eigen矩阵
    Map<MatrixXi> map1(array1, 2, 2);  // 2x2矩阵
    Map<MatrixXi> map2(array2, 2, 2);
    Map<MatrixXi> map3(array3, 2, 2);
    
    cout << "数组1映射的矩阵:\n" << map1 << endl;
    cout << "数组2映射的矩阵:\n" << map2 << endl;
    
    // 矩阵乘法
    map3 = map1 * map2;
    cout << "矩阵乘法结果:\n" << map3 << endl;
    
    // 验证原生数组也被修改
    cout << "修改后的array3: ";
    for (int i = 0; i < 4; i++) {
        cout << array3[i] << " ";
    }
    cout << endl;
    
    // 3. 使用模板函数进行矩阵乘法
    cout << "\n=== 模板函数矩阵乘法 ===" << endl;
    int arr1[4] = {1, 2, 3, 4};
    int arr2[4] = {5, 6, 7, 8};
    int arr3[4] = {0, 0, 0, 0};
    
    matrix_mul_matrix(arr1, 2, 2, arr2, 2, 2, arr3);
    
    cout << "模板函数计算结果: ";
    for (int i = 0; i < 4; i++) {
        cout << arr3[i] << " ";
    }
    cout << endl;
    
    return 0;
}

#include <Eigen/LU>
#include <chrono>

using namespace chrono;

int compare_demo() {
    cout << "=== Demo 5: 综合应用与性能优化 ===" << endl;
    
    // 1. 线性方程组求解 Ax = b
    Matrix3d A;
    Vector3d b, x;
    
    A << 1, 2, 3,
         4, 5, 6,
         7, 8, 10;
    
    b << 3, 3, 4;
    
    cout << "系数矩阵 A:\n" << A << endl;
    cout << "常数向量 b: " << b.transpose() << endl;
    
    // 使用不同的分解方法求解
    auto start = high_resolution_clock::now();
    x = A.partialPivLu().solve(b);  // 部分主元LU分解
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    
    cout << "LU分解解: " << x.transpose() << endl;
    cout << "求解时间: " << duration.count() << " 微秒" << endl;
    cout << "残差: " << (A * x - b).norm() << endl;
    
    // 2. 矩阵分解演示
    PartialPivLU<Matrix3d> lu(A);
    Matrix3d L_matrix = lu.matrixLU().triangularView<StrictlyLower>();
    Matrix3d U_matrix = lu.matrixLU().triangularView<Upper>();
    Matrix3d L_with_identity = L_matrix + Matrix3d::Identity();
    Eigen::Matrix3d P = lu.permutationP();// 获取置换矩阵P

    cout << "L矩阵（包含单位对角线）:\n" << L_with_identity << endl;
    cout << "U矩阵（显式转换）:\n" << U_matrix << endl;
    std::cout << "验证 PA = LU:\n" << P * A << std::endl;
    std::cout << "LU:\n" << L_with_identity * U_matrix << std::endl;
    cout << "验证 A = P^(-1)L*U:\n" << P.inverse()*L_with_identity * U_matrix << endl;
    cout << "原始矩阵 A:\n" << A << endl;

    // 3. 性能优化建议演示
    cout << "\n=== 性能优化演示 ===" << endl;
    
    // 固定大小矩阵 vs 动态大小矩阵
    Matrix3d fixed_mat = Matrix3d::Random();
    MatrixXd dynamic_mat = MatrixXd::Random(3, 3);
    
    // 测试固定大小矩阵运算性能
    start = high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        Matrix3d result = fixed_mat * fixed_mat;
    }
    end = high_resolution_clock::now();
    auto fixed_time = duration_cast<microseconds>(end - start);
    
    // 测试动态大小矩阵运算性能
    start = high_resolution_clock::now();
    for (int i = 0; i < 10000; ++i) {
        MatrixXd result = dynamic_mat * dynamic_mat;
    }
    end = high_resolution_clock::now();
    auto dynamic_time = duration_cast<microseconds>(end - start);
    
    cout << "固定矩阵运算时间: " << fixed_time.count() << " 微秒" << endl;
    cout << "动态矩阵运算时间: " << dynamic_time.count() << " 微秒" << endl;
    cout << "性能差异: " << (dynamic_time.count() - fixed_time.count()) << " 微秒" << endl;
    
    return 0;
}



int broadcast_array() {
    cout << "=== Demo 6: Array数组操作与广播机制 ===" << endl;
    
    // 1. Array与Matrix的区别
    Matrix3f mat1, mat2;
    Array33f arr1, arr2;
    
    mat1 << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;
    mat2 << 2, 2, 2,
            2, 2, 2,
            2, 2, 2;
    
    arr1 = mat1.array();  // Matrix转Array
    arr2 = mat2.array();
    
    cout << "矩阵 mat1:\n" << mat1 << endl;
    cout << "数组 arr1:\n" << arr1 << endl;
    
    // 2. Array的逐元素运算
    cout << "arr1 + arr2 (逐元素相加):\n" << arr1 + arr2 << endl;
    cout << "arr1 * arr2 (逐元素相乘):\n" << arr1 * arr2 << endl;
    cout << "arr1 / arr2 (逐元素相除):\n" << arr1 / arr2 << endl;
    cout << "arr1的平方:\n" << arr1.square() << endl;
    cout << "arr1的平方根:\n" << arr1.sqrt() << endl;
    cout << "arr1的指数:\n" << arr1.exp() << endl;
    cout << "arr1的对数:\n" << arr1.log() << endl;
    
    // 3. 比较操作和布尔规约,将布尔结果转换为浮点数
    Array33f bool_arr = (arr1 > 5).cast<float>();
    cout << "arr1 > 5 的布尔数组:\n" << bool_arr << endl;
    cout << "是否有元素 > 5: " << (arr1 > 5).any() << endl;
    cout << "是否所有元素 > 0: " << (arr1 > 0).all() << endl;
    cout << "大于5的元素个数: " << (arr1 > 5).count() << endl;
    
    // 4. 广播机制
    MatrixXd mat(2, 4);
    VectorXd v(2);
    
    mat << 1, 2, 6, 9,
           3, 1, 7, 2;
    v << 0, 1;
    
    cout << "原始矩阵 mat:\n" << mat << endl;
    cout << "向量 v: " << v.transpose() << endl;
    
    // 将向量v广播到矩阵的每一列
    mat.colwise() += v;
    cout << "列广播 (mat.colwise() += v):\n" << mat << endl;
    
    // 行广播示例
    RowVectorXd rv(4);
    rv << 0, 1, 2, 3;
    mat.rowwise() += rv;
    cout << "行广播 (mat.rowwise() += rv):\n" << mat << endl;
    
    // 5. 部分规约
    cout << "每列的最大值: " << mat.colwise().maxCoeff() << endl;
    cout << "每行的和: " << mat.rowwise().sum() << endl;
    cout << "每列的最小值索引: " << endl;
    VectorXd min_vals = mat.colwise().minCoeff();
    cout << min_vals.transpose() << endl;
    
    return 0;
}



#include <Eigen/Geometry>  // 几何模块


int Geometry_demo() {
    cout << "=== Demo 7: 几何变换与四元数 ===" << endl;
    
    // 1. 旋转矩阵
    Matrix3d rotation_mat = (Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitZ())  // 绕Z轴旋转45度
         * Eigen::AngleAxisd(M_PI/6, Eigen::Vector3d::UnitY())  // 绕Y轴旋转30度
         * Eigen::AngleAxisd(0, Eigen::Vector3d::UnitX())       // 绕X轴旋转0度
        ).toRotationMatrix(); // 关键：显式转换为旋转矩阵
    
    cout << "旋转矩阵:\n" << rotation_mat << endl;
    // 欧拉角
    Vector3d euler_angles = rotation_mat.eulerAngles(2, 1, 0);  // ZYX顺序
    cout << "欧拉角 (ZYX顺序, 弧度): " << euler_angles.transpose() << endl;
    cout << "欧拉角 (ZYX顺序, 角度): " << euler_angles.transpose() * 180/M_PI << endl;

    // 2. 四元数
    Quaterniond q;
    q = AngleAxisd(M_PI/4, Vector3d::UnitZ());  // 绕Z轴旋转45度的四元数
    
    cout << "四元数系数 [w, x, y, z]: [" 
         << q.w() << ", " << q.x() << ", " << q.y() << ", " << q.z() << "]" << endl;
    cout << "四元数对应的旋转矩阵:\n" << q.toRotationMatrix() << endl;
    
    // 3. 坐标变换
    Vector3d point(1, 0, 0);  // 原始点
    Vector3d rotated_point = q * point;  // 使用四元数旋转点
    
    cout << "原始点: " << point.transpose() << endl;
    cout << "旋转后的点: " << rotated_point.transpose() << endl;
    
    // 4. 仿射变换
    Affine3d transform = Translation3d(1, 2, 3)  // 平移 (1,2,3)
                        * q;                      // 旋转
    
    cout << "仿射变换矩阵:\n" << transform.matrix() << endl;
    
    Vector3d transformed_point = transform * point;
    cout << "变换后的点: " << transformed_point.transpose() << endl;

    return 0;
}




int MLE_demo() {
    cout << "=== Demo 8: 实际应用 - 最小二乘法拟合 ===" << endl;
    
    // 最小二乘法示例：拟合二次曲线 y = ax² + bx + c
    
    // 生成测试数据
    int num_points = 10;
    MatrixXd A(num_points, 3);
    VectorXd b(num_points);
    
    // 生成带噪声的二次曲线数据
    double a = 0.5, b_true = 1.0, c = 2.0;  // 真实参数
    for (int i = 0; i < num_points; ++i) {
        double x = i * 0.5;
        double y = a*x*x + b_true*x + c + 0.1 * (rand() % 100 - 50) / 50.0;  // 加噪声
        
        A(i, 0) = x * x;  // x²项
        A(i, 1) = x;      // x项
        A(i, 2) = 1;      // 常数项
        b(i) = y;
    }
    
    cout << "设计矩阵 A:\n" << A << endl;
    cout << "观测值 b: " << b.transpose() << endl;
    
    // 使用QR分解求解最小二乘问题
    Vector3d x = A.colPivHouseholderQr().solve(b);
    
    cout << "\n拟合结果:" << endl;
    cout << "真实参数: a=" << a << ", b=" << b_true << ", c=" << c << endl;
    cout << "估计参数: a=" << x(0) << ", b=" << x(1) << ", c=" << x(2) << endl;
    
    // 计算拟合误差
    VectorXd y_pred = A * x;
    double error = (y_pred - b).norm();
    cout << "拟合误差: " << error << endl;
    
    // 使用SVD分解（更稳定）
    Vector3d x_svd = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
    cout << "SVD估计参数: a=" << x_svd(0) << ", b=" << x_svd(1) << ", c=" << x_svd(2) << endl;
    
    return 0;
}