import numpy as np
import matplotlib.pyplot as plt
import math
from adjustText import adjust_text

def my_func(a):
    """Average first and last element of a 1-D array"""
    return a[1] / a[0]

data = np.loadtxt("error_tuple.txt", delimiter=",")
# print(data)
ratio = np.apply_along_axis(my_func, 1, data)
# print(ratio)
num = len(ratio)
index5 = math.floor(num * 0.05)
index25 = math.floor(num * 0.25)
index50 = math.floor(num * 0.50)
index75 = math.floor(num * 0.75)
index90 = math.floor(num * 0.90)
# print(index5, index25, index50, index75, index90)
ratio_arg = np.argsort(ratio)
mark = ratio_arg[:5].tolist()

result5 = ratio_arg[index5]
result25 = ratio_arg[index25]
result50 = ratio_arg[index50]
result75 = ratio_arg[index75]
result90 = ratio_arg[index90]
mark = [result5, result25, result50, result75, result90]

print(mark)
mark_data = data[mark]
mark_ratio = ratio[mark]
print(ratio)
print(np.sort(ratio))
x = data[:,0]
y = data[:,1]
fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.6)
ax.scatter(mark_data[:,0], mark_data[:,1])
mark_index = [5, 25, 50, 75, 90]
plt.title("Wisconsin Dataset, q = 15")
texts = [plt.text(mark_data[:,0][i], mark_data[:,1][i], str(mark_index[i]) + "%:" + str(round(mark_ratio[i], 4)), fontsize=14, color='r') for i in range(len(mark_ratio))]
adjust_text(texts,)
# for i, txt in enumerate(mark_ratio):
#     ax.annotate(str(mark_index[i]) + "%:" + str(round(txt, 4)), (mark_data[:,0][i], mark_data[:,1][i]))

plt.show()