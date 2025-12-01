
#include <glog/logging.h>  // glog头文件
#include <vector>
#include <iostream>
#include <fstream>
/* 
# 1. 基本运行
./glog_demo

# 2. 开启详细日志（级别1）
./glog_demo --v=1

# 3. 按模块设置详细级别[1,8](@ref)
./glog_demo --vmodule=main=2,other_module=1

# 4. 将日志仅输出到控制台[2,8](@ref)
./glog_demo --logtostderr=1

# 5. 设置最低日志级别[2](@ref)
./glog_demo --minloglevel=1  # 0=INFO, 1=WARNING, 2=ERROR
*/
int main(int argc, char* argv[]) {
    // 初始化glog库，argv[0]表示程序名，用于生成日志文件名[4,6](@ref)
    google::InitGoogleLogging(argv[0]);
    
    // 可选：解析命令行参数（如使用gflags库时）[2](@ref)
    // google::ParseCommandLineFlags(&argc, &argv, true);

    // === 1. 基本日志输出配置 ===[3,4](@ref)
    // 设置日志输出目录（需确保目录存在）[3](@ref)
    FLAGS_log_dir = "./logs";
    
    // 强制设置 VLOG 级别，覆盖命令行参数
    FLAGS_v = 2;  // 设置为 2，确保 VLOG(1) 和 VLOG(2) 都输出

    // 同时将日志输出到stderr（控制台）[2,8](@ref)
    FLAGS_alsologtostderr = true;
    
    // 设置控制台输出的日志颜色[3](@ref)
    FLAGS_colorlogtostderr = true;
    
    // 设置日志级别阈值（只记录该级别及以上的日志）[2,8](@ref)
    FLAGS_minloglevel = google::INFO;  // 0=INFO, 1=WARNING, 2=ERROR
    
    // 设置同时输出到stderr的日志级别阈值[2](@ref)
    FLAGS_stderrthreshold = google::WARNING;  // WARNING及以上级别输出到控制台
    
    // 设置日志文件最大大小（MB）[3](@ref)
    FLAGS_max_log_size = 100;
    
    // 磁盘满时停止日志记录[3](@ref)
    FLAGS_stop_logging_if_full_disk = true;

    // === 2. 基本日志级别输出 ===[1,4](@ref)
    LOG(INFO) << "这是一条INFO级别的日志消息";      // 普通信息
    LOG(WARNING) << "这是一条WARNING级别的日志消息"; // 警告信息
    LOG(ERROR) << "这是一条ERROR级别的日志消息";    // 错误信息
    // LOG(FATAL) << "这是一条FATAL级别的日志消息";  // 致命错误，记录后会终止程序[1](@ref)

    // === 3. 条件日志记录 ===[2,8](@ref)
    int num_cookies = 15;
    // 只有当条件满足时才记录日志[2](@ref)
    LOG_IF(INFO, num_cookies > 10) << "你有太多的cookies: " << num_cookies;
    
    // 周期性记录日志（每10次记录一次）[2](@ref)
    for (int i = 0; i < 30; ++i) {
        LOG_EVERY_N(INFO, 10) << "这是第 " << google::COUNTER << " 次记录日志";
    }
    
    // 条件+周期性记录组合[2](@ref)
    for (int i = 0; i < 20; i++) {
        LOG_IF_EVERY_N(INFO, i > 10, 5) << "i>10时每5次记录一次，当前i=" << i;
    }
    
    // 只记录前N次[2](@ref)
    for (int i = 0; i < 10; i++) {
        LOG_FIRST_N(INFO, 3) << "前3次记录：第 " << google::COUNTER << " 次";
    }

    // === 4. 检查宏（CHECK macros）===[2,8](@ref)
    int x = 10, y = 20;
    CHECK_EQ(x, 10) << "x应该等于10";  // 检查相等
    CHECK_NE(x, y) << "x不应该等于y";  // 检查不等
    CHECK_LE(x, y) << "x应该小于等于y"; // 检查小于等于
    CHECK_GT(y, x) << "y应该大于x";    // 检查大于

    // 检查失败会立即终止程序
    // CHECK_EQ(x, y)<< "x应该等于10,!=y=20";  // 检查相等

    // 检查指针非空[2](@ref)
    int* ptr = &x;
    CHECK_NOTNULL(ptr);  
    int* ptr2 = CHECK_NOTNULL(ptr);
    CHECK(ptr != nullptr) << "指针不应该为空";
    
    // 字符串检查[2](@ref)
    CHECK_STREQ("hello", "hello") << "字符串应该相等";

    // === 5. 详细日志（VLOG）===[1,8](@ref)
    // VLOG的详细级别（1-5数字越大越详细）[1](@ref)
    // 通过命令行参数--v=2来控制输出级别（默认0，不输出VLOG）
    VLOG(1) << "这是一条详细级别1的日志（需--v=1或更高）";
    VLOG(2) << "这是一条详细级别2的日志（需--v=2或更高）";
    
    // 条件VLOG[8](@ref)
    VLOG_IF(1, num_cookies > 10) << "当cookies>10时记录VLOG1";
    
    // 周期性VLOG[8](@ref)
    for (int i = 0; i < 20; i++) {
        VLOG_EVERY_N(1, 5) << "每5次记录VLOG1，当前次数: " << google::COUNTER;
    }

    // === 6. 调试模式日志（DLOG）===[2,8](@ref)
    // 只在调试模式（未定义NDEBUG宏）下生效
    DLOG(INFO) << "这条DEBUG日志只在调试模式输出";
    DLOG_IF(INFO, num_cookies > 10) << "调试模式的条件日志";

    // === 7. 错误信号处理 ===[2,8](@ref)
    // 安装失败信号处理器，程序异常时输出堆栈跟踪
    google::InstallFailureSignalHandler();
    
    // 可自定义信号处理器输出[8](@ref)
    // google::InstallFailureWriter(...);

    // === 8. 自定义日志目的地 ===[4](@ref)
    // 将不同级别的日志输出到不同文件[4](@ref)
    google::SetLogDestination(google::INFO, "./logs/info_");
    google::SetLogDestination(google::WARNING, "./logs/warning_");
    google::SetLogDestination(google::ERROR, "./logs/error_");
    google::SetLogDestination(google::FATAL, "./logs/fatal_");

    // === 9. 示例应用场景 ===
    try {
        // 模拟文件操作
        std::ofstream file("test.txt");
        if (!file.is_open()) {
            LOG(ERROR) << "无法打开文件test.txt";
        } else {
            LOG(INFO) << "文件打开成功";
            file.close();
        }
        
        // 向量操作示例
        std::vector<int> vec = {1, 2, 3};
        if (!vec.empty()) {
            VLOG(1) << "向量大小: " << vec.size();
        }
        
    } catch (const std::exception& e) {
        // 异常记录[8](@ref)
        LOG(ERROR) << "发生异常: " << e.what();
    }

    // === 10. 清理资源 ===[3,4](@ref)
    google::ShutdownGoogleLogging();
    
    std::cout << "程序执行完成，请查看./logs目录下的日志文件" << std::endl;
    return 0;
}