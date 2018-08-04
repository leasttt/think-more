import numpy as np

array = np.array([[1, 2, 3], [4, 5, 6]])
print(array)
print(array.ndim)
print(array.shape)
print(array.size)
print("######################################1")

a = np.array([2, 23, 4], dtype=np.int)  # int32, int64, float32
print(a)
print("######################################2")

b = np.zeros((3, 4))
print(b)
c = np.ones((3, 4)) * 2
print(c)
d = np.arange(10, 30, 2).reshape((2, 5))  # arange 头 尾 步长
print(d)
e = np.linspace(1, 10, 4)  # 线段
print(e)
print("######################################3")
