import importlib.util
import os.path as osp
import sys

def load(name):
    """
    使用 importlib 安全地加载场景模块，确保包上下文正确。
    """
    pathname = osp.join(osp.dirname(__file__), name)
    
    # 使用 importlib.util 现代方式加载模块
    spec = importlib.util.spec_from_file_location("", pathname)
    module = importlib.util.module_from_spec(spec)
    
    # 关键步骤：设置模块的 __package__ 属性
    module.__package__ = 'game_env.multiagent.scenarios'
    
    # 将模块添加到 sys.modules 以确保相对导入能工作
    sys.modules['game_env.multiagent.scenarios.' + osp.splitext(name)[0]] = module
    
    # 执行模块代码
    spec.loader.exec_module(module)
    return module