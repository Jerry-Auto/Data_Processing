/**
 * TensorRT C++ 推理基准测试
 *
 * 用法:
 *   ./trt_bench <model.engine> <batch_size> <iters> <warmup>
 *
 * 输出: JSON 格式耗时统计（stdout），调试信息输出到 stderr
 * TensorRT 只能在 GPU 上运行，不需要 cpu/cuda 参数
 */
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << std::endl;
    }
};

static Logger gLogger;

std::vector<char> loadEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("无法打开 Engine 文件: " + path);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), size);
    return data;
}

static std::string to_json(const std::vector<double>& times) {
    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());
    size_t n = sorted.size();
    double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    double mean = sum / n;
    double median = (n % 2 == 0) ? (sorted[n/2 - 1] + sorted[n/2]) / 2.0 : sorted[n/2];
    auto percentile = [&](double p) -> double {
        double idx = p / 100.0 * (n - 1);
        size_t lo = (size_t)std::floor(idx);
        size_t hi = (size_t)std::ceil(idx);
        if (lo == hi) return sorted[lo];
        double frac = idx - lo;
        return sorted[lo] * (1 - frac) + sorted[hi] * frac;
    };
    double sum_sq = 0.0;
    for (double t : times) sum_sq += (t - mean) * (t - mean);
    double stddev = std::sqrt(sum_sq / n);
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\"mean\":" << mean
        << ",\"median\":" << median
        << ",\"std\":" << stddev
        << ",\"min\":" << sorted.front()
        << ",\"max\":" << sorted.back()
        << ",\"p95\":" << percentile(95)
        << ",\"p99\":" << percentile(99)
        << "}";
    return oss.str();
}

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "用法: " << argv[0] << " <model.engine> <batch_size> <iters> <warmup>\n";
        return 1;
    }

    std::string engine_path = argv[1];
    int batch_size = std::atoi(argv[2]);
    int iters = std::atoi(argv[3]);
    int warmup = std::atoi(argv[4]);

    try {
        auto engine_data = loadEngine(engine_path);

        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(
            engine_data.data(), engine_data.size());
        if (!engine) {
            std::cerr << "Engine 反序列化失败\n";
            return 1;
        }
        std::cerr << "Engine 加载成功: " << engine_path << "\n";

        nvinfer1::IExecutionContext* context = engine->createExecutionContext();

        // 获取输入输出名称
        const char* input_name = nullptr;
        const char* output_name = nullptr;

        for (int i = 0; i < engine->getNbIOTensors(); i++) {
            const char* name = engine->getIOTensorName(i);
            auto mode = engine->getTensorIOMode(name);
            if (mode == nvinfer1::TensorIOMode::kINPUT)
                input_name = name;
            else
                output_name = name;
        }

        // 设置动态 shape
        nvinfer1::Dims input_dims{2, {batch_size, 4}};
        context->setInputShape(input_name, input_dims);

        auto output_dims = context->getTensorShape(output_name);
        int output_size = 1;
        for (int i = 0; i < output_dims.nbDims; i++)
            output_size *= output_dims.d[i];

        int input_size = batch_size * 4;

        // 准备输入数据
        std::vector<float> h_input(input_size);
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : h_input) v = dist(rng);

        std::vector<float> h_output(output_size);

        // GPU 内存
        void *d_input = nullptr, *d_output = nullptr;
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // 设置 tensor 地址
        context->setTensorAddress(input_name, d_input);
        context->setTensorAddress(output_name, d_output);

        // 预拷贝输入数据到 GPU（不在计时范围内）
        cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        // 单次推理 lambda（仅 enqueue + sync，不含数据搬运）
        auto run_once = [&]() {
            context->enqueueV3(stream);
            cudaStreamSynchronize(stream);
        };

        // 预热
        for (int i = 0; i < warmup; i++) run_once();

        // 正式计时
        std::vector<double> times;
        times.reserve(iters);
        for (int i = 0; i < iters; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            run_once();
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        // 输出 JSON 到 stdout
        std::cout << to_json(times) << std::endl;

        // 清理
        cudaStreamDestroy(stream);
        cudaFree(d_input);
        cudaFree(d_output);
        delete context;
        delete engine;
        delete runtime;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
