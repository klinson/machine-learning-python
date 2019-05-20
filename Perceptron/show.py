# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取文件
file = './examples.csv';
df = pd.read_csv(file, header=None);
print(df.head(10))

# 处理第4列表
y = df.loc[0: 100, 4].values;
y = np.where(y == 'Tris-setosa', -1, 1);
print(y)

# 讲第0和2列取出来分析
X = df.iloc[0: 100, [0, 2]].values;
print(X)

# 对数据可视化
plt.scatter(X[:5, 0], X[:5, 1], color='red', marker='o', label='setosa'),;
plt.scatter(X[5:10, 0], X[5:10, 1], color='blue', marker='x', label='versicolor');
plt.xlabel('花瓣长度');
plt.ylabel('花径长度');
plt.legend(loc='upper left');
plt.show();