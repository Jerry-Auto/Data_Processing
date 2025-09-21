import numpy as np 
print("\n-------------------------------\n")
A=np.arange(3,15).reshape(3,4)
print(A)
print("索引")
print(A[2][3])
print(A[2,3])
print(A[1,1:])
print("遍历输出列")
for column in A.T:
    print(column)
print("遍历输出元素")
print(A.flatten())
for item in A.flat:
    print(item)#item是地址，即迭代器
print("\n-------------------------------\n")
A=np.array([1,1,1])
B=np.array([2,2,2])
print(f"矩阵A:{A},矩阵B:{B},shape:{A.shape},{B.shape}")
C=np.vstack((A,B))
D=np.hstack((A,B))
print(f"上下堆叠矩阵C:{C},左右堆叠矩阵D:{D}")
print(f"堆叠之后的shape:{C.shape},{D.shape}")
print("\n-------------------------------\n")
print(f"矩阵A:{A},矩阵B:{B}")
print(f"加入轴之前的shape:{A.shape},{B.shape}")
A=A[np.newaxis,:]#加到第一个维度前面
B=B[np.newaxis,:]#加到第一个维度前面
print(f"矩阵A:{A},矩阵B:{B}")
print(f"加入轴之后的shape:{A.shape},{B.shape}")
E=np.vstack((A,B))
A=A[:,:,np.newaxis]#保留前两个维度，加到最后一个维度后面
B=B[:,np.newaxis]#加到了第二个维度
print(f"矩阵A:{A},矩阵B:{B}")
print(f"加入轴之后的shape:{A.shape},{B.shape}")
print(f"AB上下堆叠成的矩阵E:{E},shape:{E.shape}")
print("\n-------------------------------\n")
F=np.array([
    [5,10,12],
    [5.1,8.2,3]
    ])
print(f"矩阵F:{F},shape:{F.shape}")
#EF形状不一样就不能进行堆叠,如(2,3)和(3,3)
G=np.vstack((E,F))
print(f"EF上下堆叠成的矩阵G:{G},shape:{G.shape}")
G=G[:,np.newaxis]#加到了第二个维度，等价于G=G[:,np.newaxis]
print(f"加入轴之后的矩阵G:{G},shape:{G.shape}")
G=G[:,np.newaxis,:]
print(f"加入轴之后的矩阵G:{G},shape:{G.shape}")
print("\n-------------------------------\n")
H=np.hstack((E,F))
print(f"EF左右堆叠成的矩阵H:{H},shape:{H.shape}")
H=H[:,np.newaxis]#加到了第二个维度
print(f"加入轴之后的矩阵H:{H},shape:{H.shape}")
H=np.hstack((H,H))
print(f"H,H左右堆叠成的矩阵H:{H},shape:{H.shape}")
H=H[:,:,np.newaxis]#加到了第三个维度
print(f"加入轴之后的矩阵H:{H},shape:{H.shape}")
H=np.hstack((H,H))
print(f"H,H左右堆叠成的矩阵H:{H},shape:{H.shape}")
""" 总结：
要求除连接轴外的其他轴形状相同,维度必须相同，不会增加维度
vstack只能在第一个维度上进行堆叠 
hstack只能在第二个维度上进行堆叠
dstack深度堆叠只能在第三个维度上进行堆叠
"""
print("\n-------------------------------\n")
print("能够对张量在指定轴上进行堆叠的的方法concatenate:")
first=np.array([
    [5,10,12],
    [5.1,8.2,3],
    [5.1,8.2,3]
    ])
second=np.array([
    [5,10,12],
    [5.1,8.2,3]
    ])
first=first[np.newaxis,:]
second=second[np.newaxis,:]
print(f"矩阵first:{first},矩阵second:{second},shape:{first.shape},{second.shape}")
A_axis=np.concatenate((first,second,first,second),axis=1)
print(f"沿着轴1进行堆叠,得到矩阵A_axis:{A_axis},shape:{A_axis.shape}")
print("\n-------------------------------\n")
""" stack操作
增加一个维度，在新增的轴上进行堆叠，要求输入张量形状完全相同
numpy.stack(arrays, axis=0, out=None)
axis参数决定了堆叠的方向，对输出结果的结构有关键影响
 """
a = np.array([[1, 2], 
              [3, 4]])
b = np.array([[5, 6], 
              [7, 8]])
result_axis0 = np.stack((a,b,a,b), axis=0)
print(result_axis0.shape) # 输出 (2, 2, 2)
print(result_axis0)
print("\n-------------------------------\n")
""" 分割操作 
numpy.split(ary, indices_or_sections, axis=0)
ary:要分割的输入数组.
indices_or_sections:
    如果是​​整数​​，表示​​均分成多少份​​（必须能整除）。
    如果是​​列表/数组​​，表示​​沿轴分割的索引位置​​（如 [2, 5]表示在索引 2 和 5 处分割）.
axis:指定分割的轴（默认 axis=0,即沿行分割）。多维数组分割要指定轴​​
同样有vsplit、hsplit、dsplit，分别是沿着0,1,2轴进行分割
array_split不均等分割，多余的依次从第0轴向后均分
"""
print(f"一维数组两种分割方式")
arr = np.arange(10)  # [0 1 2 3 4 5 6 7 8 9]
result = np.split(arr, 2)  # 分成2份
print(result)  # [array([0, 1, 2, 3, 4]), array([5, 6, 7, 8, 9])]
result = np.split(arr, [3, 7])  # 在索引3和7处分割
print(result)  # [array([0, 1, 2]), array([3, 4, 5, 6]), array([7, 8, 9])]
print(f"多维数组分割")
A=np.arange(12).reshape((3,4))
B,C=np.split(A,2,axis=1)
print(f"分割前:\n{A}\n分割后:\n矩阵一:\n{B}\n矩阵二:\n{C}")
D,E,F=np.split(A,(2,3),axis=1)
print(f"分割前:\n{A}\n分割后:\n矩阵一:\n{D}\n矩阵二:\n{E}\n矩阵三:\n{F}")
print(f"其他分割方式：")
D,E,F=np.array_split(A,3,axis=1)
print(f"分割前:\n{A}\n分割后:\n矩阵一:\n{D}\n矩阵二:\n{E}\n矩阵三:\n{F}")
D,E,F=np.vsplit(A,3)
print(f"分割前:\n{A}\n分割后:\n矩阵一:\n{D}\n矩阵二:\n{E}\n矩阵三:\n{F}")
B,C=np.hsplit(A,2)
print(f"分割前:\n{A}\n分割后:\n矩阵一:\n{B}\n矩阵二:\n{C}")
arr = np.arange(11)
result = np.array_split(arr, 3)  # 分成3份，允许不等长
print(result)  #多余的依次从第0轴向后均分
print()  
print("\n-------------------------------\n")
print(f"")
print()  
print("\n-------------------------------\n")
print(f"")
print()  