from tqdm import tqdm
import time

# 外层进度条
for i in tqdm(range(5), desc="外层循环"):
    # 内层进度条
    for j in tqdm(range(100), desc="内层循环", leave=False):
        time.sleep(0.01)