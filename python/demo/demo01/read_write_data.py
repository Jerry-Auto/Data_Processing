import os
import pandas as pd
import numpy as np
import torch

script_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_path, 'data')
os.makedirs(data_path, exist_ok=True)

data_file = os.path.join(data_path, 'house.csv')
with open(data_file, 'w', encoding='UTF-8') as f:
    f.write('NumRooms,Alley,Price\n')
    f.write('NA,Pave,127500\n')
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 读取数据，并正确解析 NA
data = pd.read_csv(data_file,header=0, na_values=["NA"])

# 确保 NumRooms 是数值类型
data["NumRooms"] = pd.to_numeric(data["NumRooms"], errors="coerce")

print("原始数据:")
print(data)
print("\n数据类型:")
print(data.dtypes)

# 分离输入和输出
input_data = data.iloc[:, 0:-1]  # NumRooms, Alley
output = data.iloc[:, -1]        # Price

# 填充数值列（NumRooms）
#返回一个拷贝的DataFrame,里面是dtypes为数值的那部分DataFrame
print(input_data.select_dtypes(include=["number"]))
#提取数值列名（不会拷贝数据！）
numeric_cols = input_data.select_dtypes(include=["number"]).columns
means=input_data[numeric_cols].mean()
if len(numeric_cols) > 0:
    input_data.loc[:,numeric_cols] = input_data[numeric_cols].fillna(means)


# # 填充非数值列（Alley）
# non_numeric_cols = input_data.select_dtypes(exclude=["number"]).columns
# if len(non_numeric_cols) > 0:
#     input_data[non_numeric_cols] = input_data[non_numeric_cols].fillna("NaN")


# 对分类变量（Categorical Variables）进行独热编码（One-Hot Encoding），并额外处理缺失值（NaN）
input_data=pd.get_dummies(input_data,dummy_na=True)

print("\n填充后的输入数据:")
print(input_data)

X=torch.from_numpy(input_data.to_numpy(dtype=float))
Y=torch.as_tensor(output.values.tolist(),dtype=torch.float64)

print(X)
print(Y)