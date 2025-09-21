import numpy as np 
cars=np.array([
    [5,10,12,6],
    [5.1,8.2,3,6],
    [5,5,3,6]
    ],dtype=np.float32)
print("数据:",cars,"\n维度:",cars.ndim)
print("\n-------------------------------\n")


test1 = np.array([5, 10, 12, 6])
test2 = np.array([5.1, 8.2, 11, 6.3])

# 首先需要把它们都变成二维，下面这两种方法都可以加维度
test1 = np.expand_dims(test1, 0)
test2 = test2[np.newaxis, :]

print("test1加维度后 ", test1)
print("test2加维度后 ", test2)

# 然后再在第一个维度上叠加
all_tests = np.concatenate([test1, test2])
print("括展后\n", all_tests)
 
print("\n-------------------------------\n")
zero_array=np.zeros((3,4),dtype=np.int16)
print(zero_array)

print("\n-------------------------------\n")
ones_array=np.ones((3,4),dtype=np.float32)
print(ones_array)

print("\n-------------------------------\n")
empty=np.empty((3,4))
print(empty)

print("\n-------------------------------\n")
arrange=np.arange(10,22,1).reshape((3,4))
print(arrange)
print("\n")
arrange=arrange.reshape((2,6))
print(arrange)
print("\n-------------------------------\n")
linspace=np.linspace(1,10,5)
print(linspace)

print("\n-------------------------------\n")
""" numpy赋值相当于c++引用，同一份数据，不同的名字 """
a=np.arange(4,dtype=np.float16)
b=a
c=b
a[0]=0.5
print(f"{a},{b},{c}")
print(f"深拷贝方法：")
b=a.copy()
b[1:3]=[15,16]
print(f"{a},{b},{c}")
