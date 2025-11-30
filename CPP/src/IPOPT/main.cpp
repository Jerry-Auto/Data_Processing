#include"IPOPT_demo.h"
#include<iostream>

int main(){
    using namespace std;
    cout << "CppAD和IPOPT优化求解器演示程序!" << endl;
    cout << "==================================" << endl;
    cout << "问题描述:" << endl;
    cout << "最小化: x1*x4*(x1+x2+x3) + x3" << endl;
    cout << "约束条件:" << endl; 
    cout << "  x1*x2*x3*x4 ≥ 25" << endl;
    cout << "  x1² + x2² + x3² + x4² = 40" << endl;
    cout << "  1.0 ≤ x1,x2,x3,x4 ≤ 5.0" << endl;
    cout << "==================================" << endl;
    
    // 调用优化求解函数
    bool result = get_started();
    
    // 根据求解结果返回退出码
    if (result) {
        return 0;  // 成功退出
    } else {
        return 1;  // 失败退出
    }
    return 0;
}