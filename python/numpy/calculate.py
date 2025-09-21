import numpy as np 
print("\n-------------------------------\n")
print("元素操作")
""" 可以是矩阵，都是对里面的元素做运算 """
a=np.array([10,20,30,40])
b=np.arange(5,9)

print(a,b)
add=a+b
subtract=a-b
multiply=a*b#相当于向量内积
divide=a/b
print("+-*/")
print(add,subtract,multiply,divide)
print("次方")
power=b**2
print(power)
print("三角函数")
sin=10*np.sin(a)
cos=20*np.cos(b)
tan=30*np.tan(a)
print(sin,cos,tan)
print("比较：")
print(a>=30)#==,< <= > >=
print("\n-------------------------------\n")
a=a.reshape(2,2)
b=b.reshape(2,2)
""" 矩阵运算 """
print("矩阵运算")
print(a)
print(b)
print("乘法")
print(a.dot(b))
print(np.dot(a,b))
print("\n-------------------------------\n")
a=np.random.random((2,4))
print("寻找矩阵中的特定值")
print(a)
print(np.sum(a))
print(np.min(a))
print(np.max(a))
print(np.mean(a))
print("\n-------------------------------\n")
print("numpy的axis")
high_dim=np.random.random((2,2,3,4))
print(f"数组维度(axis数量):{high_dim.ndim}")
print(high_dim)
print("对特定轴操作")
print(np.sum(high_dim,axis=3))
#print(np.sum(high_dim,axis=1))

print("\n-------------------------------\n")
A=np.arange(10,-2,-1).reshape(3,4)
print(A)
print("差分")
print(np.diff(A))
print("均值")
print(np.average(A))
print(A.mean())
print("累积求和序列")
print(np.cumsum(A))
print("中位数")
print(np.median(A))
print("非0数的行与列")
print(np.nonzero(A))
print("排序")
print(np.sort(A))
print("矩阵转置")
print(np.transpose(A))
print(A.T)
print("(A^T)*A")
print((A.T).dot(A))
print("限制最大最小")
print(np.clip(A,2,6))











