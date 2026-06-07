/**
 * TensorRT C++ 推理示例
 *
 * 用法:
 *   ./trt_demo <model.engine 路径>
 *
 * 前置条件:
 *   已安装 TensorRT 并构建 Engine
 */
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include <numeric>

// TensorRT 日志器
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << std::endl;
    }
};

static Logger gLogger;

// 读取 Engine 文件
std::vector<char> loadEngineFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开 Engine 文件: " + path);
    }
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);
    return data;
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "用法: " << argv[0] << " <model.engine 路径>\n";
        return 1;
    }
    const std::string engine_path = argv[1];

    // ========== 1. 加载 Engine ==========
    auto engine_data = loadEngineFile(engine_path);

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(
        engine_data.data(), engine_data.size());

    if (!engine) {
        std::cerr << "Engine 反序列化失败\n";
        return 1;
    }
    std::cout << "Engine 加载成功: " << engine_path << "\n";

    // ========== 2. 创建执行上下文 ==========
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();

    // ========== 3. 获取输入输出信息 ==========
    const char* input_name = nullptr;
    const char* output_name = nullptr;
    int input_idx = -1, output_idx = -1;

    for (int i = 0; i < engine->getNbIOTensors(); i++) {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name = name;
            input_idx = i;
        } else {
            output_name = name;
            output_idx = i;
        }
    }
    std::cout << "输入名: " << input_name << "\n";
    std::cout << "输出名: " << output_name << "\n";

    // ========== 4. 设置输入 shape 并分配内存 ==========
    const int batch_size = 2;
    const int input_dim = 4;
    const int input_size = batch_size * input_dim;

    // 设置动态 shape
    nvinfer1::Dims input_dims{2, {batch_size, input_dim}};
    context->setInputShape(input_name, input_dims);

    // 获取实际输出 shape
    auto output_dims = context->getTensorShape(output_name);
    int output_size = 1;
    for (int i = 0; i < output_dims.nbDims; i++) {
        output_size *= output_dims.d[i];
    }
    std::cout << "输出 shape: [";
    for (int i = 0; i < output_dims.nbDims; i++) {
        std::cout << output_dims.d[i];
        if (i < output_dims.nbDims - 1) std::cout << ", ";
    }
    std::cout << "]\n";

    // Host 内存
    std::vector<float> h_input(input_size);
    std::vector<float> h_output(output_size);

    // 填充输入数据
    float input_vals[] = {1.0f, 2.0f, 3.0f, 4.0f,
                          0.5f, 1.5f, 2.5f, 3.5f};
    std::memcpy(h_input.data(), input_vals, input_size * sizeof(float));

    std::cout << "\n输入数据:\n";
    for (int i = 0; i < input_size; i++) {
        std::cout << h_input[i] << " ";
        if ((i + 1) % input_dim == 0) std::cout << "\n";
    }

    // GPU 内存
    void *d_input = nullptr, *d_output = nullptr;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));

    // ========== 5. 拷贝输入并推理 ==========
    cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float),
               cudaMemcpyHostToDevice);

    // 设置 tensor 地址
    context->setTensorAddress(input_name, d_input);
    context->setTensorAddress(output_name, d_output);

    // 异步推理
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    context->enqueueV3(stream);
    cudaStreamSynchronize(stream);

    // 拷贝输出
    cudaMemcpy(h_output.data(), d_output, output_size * sizeof(float),
               cudaMemcpyDeviceToHost);

    // ========== 6. 打印输出 ==========
    std::cout << "\n输出数据:\n";
    int cols = output_dims.d[output_dims.nbDims - 1];
    for (int i = 0; i < output_size; i++) {
        std::cout << h_output[i] << " ";
        if ((i + 1) % cols == 0) std::cout << "\n";
    }

    // ========== 7. 多次推理 ==========
    std::cout << "\n--- 连续 3 次推理 ---\n";
    for (int iter = 0; iter < 3; iter++) {
        // 随机输入
        for (int i = 0; i < input_dim; i++) {
            h_input[i] = static_cast<float>(rand()) / RAND_MAX;
        }
        cudaMemcpy(d_input, h_input.data(), input_dim * sizeof(float),
                   cudaMemcpyHostToDevice);

        // 单 batch
        nvinfer1::Dims single_dims{2, {1, input_dim}};
        context->setInputShape(input_name, single_dims);
        context->setTensorAddress(input_name, d_input);
        context->setTensorAddress(output_name, d_output);
        context->enqueueV3(stream);
        cudaStreamSynchronize(stream);

        auto out_dims = context->getTensorShape(output_name);
        int out_size = 1;
        for (int i = 0; i < out_dims.nbDims; i++) out_size *= out_dims.d[i];

        std::vector<float> single_out(out_size);
        cudaMemcpy(single_out.data(), d_output, out_size * sizeof(float),
                   cudaMemcpyDeviceToHost);

        std::cout << "  iter " << iter << ": ";
        for (int i = 0; i < out_size; i++) std::cout << single_out[i] << " ";
        std::cout << "\n";
    }

    // ========== 8. 清理 ==========
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_output);
    delete context;
    delete engine;
    delete runtime;

    std::cout << "\n资源已释放，完成。\n";
    return 0;
}
