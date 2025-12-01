#include"eigen_learn.h"
#include<iostream>

int main(){
    using std::cin;
    using std::cout;
    char choice;
    cout<<"请输入演示的示例选项：\n";
    cout<<"A:综合示例\t B:矩阵定义与初始化\t C:矩阵运算与代数操作\t D:块操作与切片\n"
    <<"E:特征值分解与Map类\t F:综合应用与性能优化\t G:Array数组操作与广播机制\n"
    <<"H:几何变换与四元数 \t I:最小二乘法拟合\t 退出:Q \n";
    while(std::cin>>choice&& toupper(choice) != 'Q'){
        while (cin.get() != '\n')
        continue;
        switch (toupper(choice))
        {
            case 'A':simple_demo();
            break;
            case 'B':init_matrix();
            break;
            case 'C':matrix_operate();
            break;
            case 'D':block_col_row();
            break;
            case 'E':eigenvalue_map();
            break;
            case 'F':compare_demo();
            break;
            case 'G':broadcast_array();
            break;
            case 'H':Geometry_demo();
            break;
            case 'I':MLE_demo();
            break;
        }
        cout<<"请输入演示的示例选项：\n";
        cout<<"A:综合示例\t B:矩阵定义与初始化\t C:矩阵运算与代数操作\t D:块操作与切片\n"
        <<"E:特征值分解与Map类\t F:综合应用与性能优化\t G:Array数组操作与广播机制\n"
        <<"H:几何变换与四元数 \t I:最小二乘法拟合\t 退出:Q \n";
    }
    cout << "程序结束\n";
    
    return 0;
}