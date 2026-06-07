/**
 * ONNX Runtime C++ 推理示例
 *
 * 用法:
 *   ./onnx_demo <model.onnx 路径>
 */
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <vector>
#include <numeric>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "用法: " << argv[0] << " <model.onnx 路径>\n";
        return 1;
    }
    const std::string model_path = argv[1];

    // ========== 1. 初始化环境 ==========
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_demo");

    // 会话选项
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // ========== 2. 创建会话 ==========
    Ort::Session session(env, model_path.c_str(), session_options);
    std::cout << "模型加载成功: " << model_path << "\n";

    // ========== 3. 获取输入输出名称 ==========
    Ort::AllocatorWithDefaultOptions allocator;

    // 输入名称
    size_t num_inputs = session.GetInputCount();
    std::vector<std::string> input_name_strs(num_inputs);
    std::vector<const char*> input_names(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
        auto name = session.GetInputNameAllocated(i, allocator);
        input_name_strs[i] = name.get();
        input_names[i] = input_name_strs[i].c_str();
    }

    // 输出名称
    size_t num_outputs = session.GetOutputCount();
    std::vector<std::string> output_name_strs(num_outputs);
    std::vector<const char*> output_names(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        output_name_strs[i] = name.get();
        output_names[i] = output_name_strs[i].c_str();
    }

    std::cout << "输入: " << input_name_strs[0] << "\n";
    std::cout << "输出: " << output_name_strs[0] << "\n";

    // ========== 4. 构造输入 ==========
    // batch_size=2, input_dim=4
    std::vector<int64_t> input_shape = {2, 4};
    std::vector<float> input_data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        0.5f, 1.5f, 2.5f, 3.5f
    };

    std::cout << "\n输入数据:\n";
    for (size_t i = 0; i < input_data.size(); i++) {
        std::cout << input_data[i] << " ";
        if ((i + 1) % 4 == 0) std::cout << "\n";
    }

    // 创建输入 tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    // ========== 5. 推理 ==========
    std::vector<Ort::Value> input_values;
    input_values.push_back(std::move(input_tensor));

    auto output_values = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        input_values.data(),
        input_values.size(),
        output_names.data(),
        output_names.size()
    );

    // ========== 6. 取输出 ==========
    auto& output_tensor = output_values.front();
    auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
    auto output_shape = type_info.GetShape();
    size_t output_elements = type_info.GetElementCount();
    float* output_data = output_tensor.GetTensorMutableData<float>();

    std::cout << "\n输出 shape: [";
    for (size_t i = 0; i < output_shape.size(); i++) {
        std::cout << output_shape[i];
        if (i < output_shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    std::cout << "输出数据:\n";
    int cols = output_shape.back();
    for (size_t i = 0; i < output_elements; i++) {
        std::cout << output_data[i] << " ";
        if ((i + 1) % cols == 0) std::cout << "\n";
    }

    // ========== 7. 多次推理演示 ==========
    std::cout << "\n--- 连续 3 次推理 ---\n";
    for (int iter = 0; iter < 3; iter++) {
        std::vector<float> rand_data(4);
        for (auto& v : rand_data) v = static_cast<float>(rand()) / RAND_MAX;

        std::vector<int64_t> shape = {1, 4};
        Ort::Value t = Ort::Value::CreateTensor<float>(
            memory_info, rand_data.data(), rand_data.size(),
            shape.data(), shape.size());

        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(t));
        auto outs = session.Run(Ort::RunOptions{nullptr},
                                input_names.data(), inputs.data(), 1,
                                output_names.data(), 1);

        float* out = outs[0].GetTensorMutableData<float>();
        std::cout << "  iter " << iter << ": ";
        for (size_t i = 0; i < outs[0].GetTensorTypeAndShapeInfo().GetElementCount(); i++) {
            std::cout << out[i] << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
