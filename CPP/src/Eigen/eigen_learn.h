#pragma once

#include <Eigen/Dense>
 
// 通用的矩阵乘法模板函数（支持行优先和列优先）
template <typename T>
static void matrix_mul_matrix(T* p1, int iRow1, int iCol1, 
                             T* p2, int iRow2, int iCol2, 
                             T* p3) {
    if (iCol1 != iRow2) return;  // 修正：应该是iCol1 != iRow2
 
    // 行优先存储顺序
    Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > 
        map1(p1, iRow1, iCol1);
    Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > 
        map2(p2, iRow2, iCol2);
    Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > 
        map3(p3, iRow1, iCol2);  // 修正：结果矩阵大小应为iRow1 x iCol2
 
    map3 = map1 * map2;
}

int simple_demo();
int init_matrix() ;
int matrix_operate();
int block_col_row();
int eigenvalue_map();
int compare_demo();
int broadcast_array();
int Geometry_demo();
int MLE_demo();