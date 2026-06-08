/**
 * ONNX Runtime C++ 推理基准测试
 *
 * 用法:
 *   ./onnx_bench <model.onnx> <batch_size> <iters> <warmup> <cpu|cuda>
 *
 * 输出: JSON 格式耗时统计（stdout），调试信息输出到 stderr
 */
#include <onnxruntime_cxx_api.h>
#ifdef ORT_CUDA_AVAILABLE
#include <cuda_runtime_api.h>
#endif
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

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
    if (argc != 6) {
        std::cerr << "用法: " << argv[0] << " <model.onnx> <batch_size> <iters> <warmup> <cpu|cuda>\n";
        return 1;
    }

    std::string model_path = argv[1];
    int batch_size = std::atoi(argv[2]);
    int iters = std::atoi(argv[3]);
    int warmup = std::atoi(argv[4]);
    std::string device = argv[5];

    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "onnx_bench");
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        bool use_cuda = false;
        if (device == "cuda") {
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                use_cuda = true;
                std::cerr << "使用 CUDAExecutionProvider\n";
            } catch (const std::exception& e) {
                std::cerr << "CUDAExecutionProvider 不可用，回退到 CPU: " << e.what() << "\n";
            }
        }
        if (!use_cuda) {
            std::cerr << "使用 CPUExecutionProvider\n";
        }

        Ort::Session session(env, model_path.c_str(), session_options);

        // 获取输入输出名称
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
        auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
        std::string input_name_str = input_name_alloc.get();
        std::string output_name_str = output_name_alloc.get();
        const char* input_names[] = { input_name_str.c_str() };
        const char* output_names[] = { output_name_str.c_str() };

        // 准备输入数据
        std::vector<int64_t> input_shape = { batch_size, 4 };
        std::vector<float> input_data(batch_size * 4);
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto& v : input_data) v = dist(rng);

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // 单次推理 lambda
        auto run_once = [&]() {
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_data.data(), input_data.size(),
                input_shape.data(), input_shape.size());
            auto outputs = session.Run(Ort::RunOptions{nullptr},
                                       input_names, &input_tensor, 1,
                                       output_names, 1);
#ifdef ORT_CUDA_AVAILABLE
            if (use_cuda) cudaDeviceSynchronize();
#endif
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

    } catch (const Ort::Exception& e) {
        std::cerr << "ORT 错误: " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
