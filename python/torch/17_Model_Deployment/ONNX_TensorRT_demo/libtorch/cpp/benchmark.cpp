/**
 * LibTorch C++ 推理基准测试
 *
 * 用法:
 *   ./libtorch_bench <model.pt> <batch_size> <iters> <warmup> <cpu|cuda>
 *
 * 输出: JSON 格式耗时统计（stdout），调试信息输出到 stderr
 */
#include <torch/script.h>
#ifdef AT_CUDA
#include <ATen/cuda/CUDAContext.h>
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
        std::cerr << "用法: " << argv[0] << " <model.pt> <batch_size> <iters> <warmup> <cpu|cuda>\n";
        return 1;
    }

    std::string model_path = argv[1];
    int batch_size = std::atoi(argv[2]);
    int iters = std::atoi(argv[3]);
    int warmup = std::atoi(argv[4]);
    std::string device = argv[5];

    try {
        // 选择设备
        torch::Device torch_device(torch::kCPU);
        if (device == "cuda") {
#ifdef AT_CUDA
            if (!at::cuda::is_available()) {
                std::cerr << "CUDA 不可用，回退到 CPU\n";
                device = "cpu";
            } else {
                torch_device = torch::Device(torch::kCUDA);
            }
#else
            std::cerr << "编译时未启用 CUDA，回退到 CPU\n";
            device = "cpu";
#endif
        }
        std::cerr << "使用设备: " << device << "\n";

        // 加载模型
        torch::jit::script::Module module;
        module = torch::jit::load(model_path);
        module.to(torch_device);
        module.eval();

        // 准备输入数据
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> input_data(batch_size * 4);
        for (auto& v : input_data) v = dist(rng);

        torch::Tensor input = torch::from_blob(
            input_data.data(), {batch_size, 4}, torch::kFloat32
        ).clone().to(torch_device);

        // 单次推理 lambda
        auto run_once = [&]() {
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input);
            torch::NoGradGuard no_grad;
            auto output = module.forward(inputs).toTensor();
#ifdef AT_CUDA
            if (torch_device.is_cuda()) {
                c10::cuda::getCurrentCUDAStream().synchronize();
            }
#endif
        };

        // 预热
        for (int i = 0; i < warmup; i++) run_once();

        // 正式计时
        std::vector<double> times;
        times.reserve(iters);
        for (int i = 0; i < iters; i++) {
#ifdef AT_CUDA
            if (torch_device.is_cuda()) {
                c10::cuda::getCurrentCUDAStream().synchronize();
            }
#endif
            auto t0 = std::chrono::high_resolution_clock::now();
            run_once();
#ifdef AT_CUDA
            if (torch_device.is_cuda()) {
                c10::cuda::getCurrentCUDAStream().synchronize();
            }
#endif
            auto t1 = std::chrono::high_resolution_clock::now();
            times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        // 输出 JSON 到 stdout
        std::cout << to_json(times) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
