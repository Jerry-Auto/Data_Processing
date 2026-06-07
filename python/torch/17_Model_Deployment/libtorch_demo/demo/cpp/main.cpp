/**
 * C++ 端：加载 TorchScript 模型 (.pt) 并执行推理
 *
 * 用法:
 *   ./libtorch_demo <model.pt 路径>
 * 例如:
 *   ./libtorch_demo ../../pytorch/model.pt
 */
#include <torch/script.h>  // libtorch 的唯一头文件，包含一切

#include <iostream>
#include <vector>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "用法: " << argv[0] << " <model.pt 路径>\n";
        return 1;
    }

    const std::string model_path = argv[1];

    // ========== 1. 加载模型 ==========
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
        std::cout << "模型加载成功: " << model_path << "\n";
    } catch (const c10::Error& e) {
        std::cerr << "模型加载失败: " << e.what() << "\n";
        return 1;
    }

    // 切到 eval 模式
    module.eval();

    // ========== 2. 构造输入 ==========
    // 对应 Python 端: input_dim=4, batch_size=2
    // torch::ones / torch::randn 都可以，这里用手动赋值便于观察输出
    torch::Tensor input = torch::tensor(
        {{1.0f, 2.0f, 3.0f, 4.0f},
         {0.5f, 1.5f, 2.5f, 3.5f}}
    );
    std::cout << "\n输入 tensor:\n" << input << "\n";
    std::cout << "输入 shape: " << input.sizes() << "\n";

    // ========== 3. 推理 ==========
    // forward() 接受 vector<IValue>，返回 IValue
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    torch::NoGradGuard no_grad;  // 推理时关闭梯度计算
    torch::Tensor output = module.forward(inputs).toTensor();

    std::cout << "\n输出 tensor:\n" << output << "\n";
    std::cout << "输出 shape: " << output.sizes() << "\n";

    // ========== 4. 逐元素访问示例 ==========
    std::cout << "\n逐行访问输出:\n";
    for (int64_t i = 0; i < output.size(0); i++) {
        std::cout << "  样本 " << i << ": ";
        for (int64_t j = 0; j < output.size(1); j++) {
            std::cout << output[i][j].item<float>() << " ";
        }
        std::cout << "\n";
    }

    // ========== 5. 多次推理（模拟批量请求）==========
    std::cout << "\n--- 连续 3 次推理 ---\n";
    for (int iter = 0; iter < 3; iter++) {
        torch::Tensor rand_input = torch::randn({1, 4});
        std::vector<torch::jit::IValue> batch_inputs = {rand_input};
        torch::Tensor out = module.forward(batch_inputs).toTensor();
        std::cout << "  iter " << iter << " | 输出: " << out.squeeze() << "\n";
    }

    return 0;
}