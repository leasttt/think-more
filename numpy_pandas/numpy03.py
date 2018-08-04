import numpy as np

a = np.arange(3, 15).reshape((3, 4))
print(a)
print(a[2][3])  # 第2行 第3列
print(a[2, 3])
print(a[2, :])  # 第2行所有数
print(a[2, 1:3])  # 第2行的1,2数
print("######################################1")

for row in a:
    print(row)  # 迭代行

for column in a.T:
    print(column)  # 迭代列

print(a.flatten)  # 碾平成一行的迭代
for item in a.flat:
    print(item)  # 迭代元素项
print("######################################2")

a = np.ones((3))
b = np.ones((3)) * 2
print(a)
print(b)

print(np.hstack((a, b)))  # 横向合并
print(np.vstack((a, b)))  # 竖向合并

print(a[np.newaxis, :])  # 增加行维度
print(a[:, np.newaxis])  # 增加列维度

print(np.hstack((a[:, np.newaxis], b[:, np.newaxis])))
print(np.vstack((a[:, np.newaxis], b[:, np.newaxis])))
print("######################################3")

a = np.ones((1, 3))
b = np.ones((1, 3)) * 2
print(np.concatenate((a, b, b, a), axis=0))
print("######################################4")

a = np.arange(12).reshape((3, 4))
print(a)
print(np.split(a, 3, axis=0))  # 按行分成3块
print(np.split(a, 2, axis=1))  # 按列分成2块

print(np.array_split(a, 3, axis=1))  # 不等分割
print(np.vsplit(a, 3))
print("######################################5")
