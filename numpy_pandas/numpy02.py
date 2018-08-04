import numpy as np

a = np.array([10, 20, 30, 40])
b = np.arange(1, 5)
print(a)
print(b)
print("######################################1")

print(a - b)
print(a + b)
print(a * b)
print(a / b)
print(b ** 2)
print(10 * np.sin(a))
print(b < 3)
print(b == 3)
print("######################################2")

a = np.array([[1, 1], [0, 1]])
b = np.arange(1, 5).reshape((2, 2))
print(a)
print(b)
print("######################################3")

print(a * b)  # 逐个相乘
print(np.dot(a, b))  # 矩阵相乘
print(a.dot(b))  # 矩阵相乘
print("######################################4")

a = np.random.randint(0, 10, (2, 4))  # min max shape
print(a)
print(np.sum(a))
print(np.min(a))
print(np.max(a))
print(np.sum(a, axis=0))  # axis=0表示index从行1到行n
print(np.min(a, axis=1))  # axis=1表示index从列1到列n
print(np.max(a, axis=1))
print("######################################5")

a = np.arange(14, 2, -1).reshape((3, 4))
print(a)
print(np.argmin(a))  # 最小值的index
print(np.argmax(a))  # 最大值的index
print(np.mean(a, axis=0))  # 平均值
print(np.median(a, axis=1))  # 中位数
print(np.cumsum(a, axis=0))  # 累加
print(np.diff(a))  # 累差
print(np.nonzero(a))  # 非0的index列表
print(np.sort(a, axis=1))  # 排序,默认axis=1
print(np.transpose(a))  # 转置
print(a.T)  # 转置
print(a.T.dot(a))  # 矩阵转置相乘
print(np.clip(a, 5, 9))  # 小于5的变5,大于9的变9
