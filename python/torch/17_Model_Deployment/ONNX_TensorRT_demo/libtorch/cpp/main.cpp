/**
 * LibTorch C++ 推理示例（支持 CPU 和 GPU）
 *
 * 用法:
 *   ./libtorch_demo <model.pt 路径> [cpu|cuda]
 *
 * 例如:
 *   ./libtorch_demo ../../model/model.pt cpu
 *   ./libtorch_demo ../../model/model.pt cuda
 */
#include <torch/script.h>  // libtorch 的唯一头文件，包含一切
#ifdef AT_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif

#include <iostream>
#include <vector>
#include <string>

int main(int argc, const char* argv[]) {
    if (argc < 2 || argc > 3) {
        std::cerr << "用法: " << argv[0] << " <model.pt 路径> [cpu|cuda]\n";
        std::cerr << "  默认设备: cpu\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string device_str = (argc == 3) ? argv[2] : "cpu";

    // ========== 1. 选择设备 ==========
    torch::Device device(torch::kCPU);
    if (device_str == "cuda") {
#ifdef AT_CUDA
        if (!at::cuda::is_available()) {
            std::cerr << "错误: CUDA 不可用\n";
            return 1;
        }
        device = torch::Device(torch::kCUDA, 0);
        std::cout << "使用设备: CUDA (GPU 0)\n";
#else
        std::cerr << "错误: 编译时未启用 CUDA 支持\n";
        return 1;
#endif
    } else {
        std::cout << "使用设备: CPU\n";
    }

    // ========== 2. 加载模型 ==========
    torch::jit::script::Module module;
    try {
        module = torch::jit::load(model_path);
        std::cout << "模型加载成功: " << model_path << "\n";
    } catch (const c10::Error& e) {
        std::cerr << "模型加载失败: " << e.what() << "\n";
        return 1;
    }

    // 将模型移动到指定设备
    module.to(device);

    // 切到 eval 模式
    module.eval();

    // ========== 3. 构造输入 ==========
    // 对应 Python 端: input_dim=4, batch_size=2
    // torch::ones / torch::randn 都可以，这里用手动赋值便于观察输出
    torch::Tensor input = torch::tensor(
        {{1.0f, 2.0f, 3.0f, 4.0f},
         {0.5f, 1.5f, 2.5f, 3.5f}}
    );

    // 将输入移动到指定设备
    input = input.to(device);

    std::cout << "\n输入 tensor:\n" << input << "\n";
    std::cout << "输入 shape: " << input.sizes() << "\n";
    std::cout << "输入设备: " << input.device() << "\n";

    // ========== 4. 推理 ==========
    // forward() 接受 vector<IValue>，返回 IValue
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    torch::NoGradGuard no_grad;  // 推理时关闭梯度计算
    torch::Tensor output = module.forward(inputs).toTensor();

    // 如果是 GPU，同步确保计算完成
#ifdef AT_CUDA
    if (device.is_cuda()) {
        c10::cuda::getCurrentCUDAStream().synchronize();
    }
#endif

    std::cout << "\n输出 tensor:\n" << output << "\n";
    std::cout << "输出 shape: " << output.sizes() << "\n";
    std::cout << "输出设备: " << output.device() << "\n";

    // ========== 5. 逐元素访问示例 ==========
    // 注意：如果在 GPU 上，需要先移回 CPU 才能逐元素访问
    torch::Tensor output_cpu = output.cpu();
    std::cout << "\n逐行访问输出:\n";
    for (int64_t i = 0; i < output_cpu.size(0); i++) {
        std::cout << "  样本 " << i << ": ";
        for (int64_t j = 0; j < output_cpu.size(1); j++) {
            std::cout << output_cpu[i][j].item<float>() << " ";
        }
        std::cout << "\n";
    }

    // ========== 6. 多次推理（模拟批量请求）==========
    std::cout << "\n--- 连续 3 次推理 ---\n";
    for (int iter = 0; iter < 3; iter++) {
        torch::Tensor rand_input = torch::randn({1, 4}).to(device);
        std::vector<torch::jit::IValue> batch_inputs = {rand_input};
        torch::Tensor out = module.forward(batch_inputs).toTensor();

#ifdef AT_CUDA
        if (device.is_cuda()) {
            c10::cuda::getCurrentCUDAStream().synchronize();
        }
#endif

        std::cout << "  iter " << iter << " | 输出: " << out.cpu().squeeze() << "\n";
    }

    return 0;
}
