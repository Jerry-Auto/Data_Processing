#pragma once

#include <Eigen/Dense>
 
template <typename T>
static void matrix_mul_matrix(T* p1, int iRow1, int iCol1, T* p2, int iRow2, int iCol2, T* p3)
{
    if (iRow1 != iRow2) return;
 
    //列优先
    //Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > map1(p1, iRow1, iCol1);
    //Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > map2(p2, iRow2, iCol2);
    //Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> > map3(p3, iCol1, iCol2);
 
    //行优先
    Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > map1(p1, iRow1, iCol1);
    Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > map2(p2, iRow2, iCol2);
    Eigen::Map< Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > map3(p3, iCol1, iCol2);
 
    map3 = map1 * map2;
};

int demo();