import argparse
import os
import json
import yaml
from types import SimpleNamespace

def dict_to_object(d):
    """
    一个极其优雅的辅助函数：将嵌套字典转换为可以通过点号 (.) 访问的对象
    例如：config['train']['lr'] -> config.train.lr
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_object(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_object(i) for i in d]
    return d

def main():
    # 1. 配置 argparse 命令行解析器
    parser = argparse.ArgumentParser(description="Multimodal Project Configuration Loader")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=True, # 设为必填，引导你尝试不同的配置文件
        help="Path to the config file (either .json or .yaml)"
    )
    args = parser.parse_args()

    # 2. 获取文件后缀名
    file_ext = os.path.splitext(args.config)[1].lower()
    raw_config_dict = {}

    # 3. 根据不同的文件后缀，选择对应的读取方式
    if file_ext == ".json":
        print(f"⚙️ 检查到 [.json] 后缀，正在使用 Python 内置 json 库解析...")
        with open(args.config, "r", encoding="utf-8") as f:
            raw_config_dict = json.load(f)
            
    elif file_ext in [".yaml", ".yml"]:
        print(f"⚙️ 检查到 [{file_ext}] 后缀，正在使用 PyYAML 库解析...")
        with open(args.config, "r", encoding="utf-8") as f:
            raw_config_dict = yaml.safe_load(f)
            
    else:
        raise ValueError(f"❌ 不支持的文件格式: {file_ext}。请提供 .json 或 .yaml 文件。")

    # 4. 将字典转换为对象结构
    config = dict_to_object(raw_config_dict)

    # 5. 模拟进入核心业务逻辑（感受点号访问的丝滑）
    print("\n" + "="*50)
    print(f" 🚀 项目 [{config.project_name} v{config.version}] 启动成功！")
    print("="*50)
    
    print(f"➔ [模型配置] 加载模型: {config.model.name} (维度: {config.model.embed_dim})")
    print(f"➔ [数据配置] 数据集列表: {config.data.dataset_names} (线程数: {config.data.num_workers})")
    print(f"➔ [训练配置] 运行设备: {config.train.device} | 初始学习率: {config.train.lr}")
    
    print("="*50)
    print("... 开始构建 PyTorch DataLoader 和分布式训练流程 ...\n")

if __name__ == "__main__":
    main()