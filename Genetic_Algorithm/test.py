import numpy as np

# 创建一个numpy数组
arr = np.array([[3, 1, 4], [1, 5, 9], [2, 6, 5], [3, 5, 3]])

# 使用numpy的argsort函数对整个数组进行排序
sorted_indices = np.argsort(arr[:, -1])

# 根据最后一个数字进行排序
last_numbers = arr[:, -1]
sorted_indices = sorted_indices[::-1] if last_numbers[-1] % 2 == 0 else sorted_indices

# 输出排序后的数组
sorted_arr = arr[sorted_indices]
print(sorted_arr)
