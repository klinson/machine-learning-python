# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from Perceptron import Perceptron


def plot_decision_regions(X, y, classifier, resolution=0.02):
	#画图划线分割

	markers = ['s', 'x', 'o', 'v'];
	colors= ['red', 'blue', 'lightred', 'gray', 'cyan']
	cmap = ListedColormap(colors[:len(np.unique(y))]);

	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max();
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max();

	#print(x1_min, x1_max, x2_min, x2_max);

	# 根据数据最大最小值构建向量,差值resolution
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
	#print(np.arange(x1_min, x1_max, resolution).shape, np.arange(x1_min, x1_max, resolution), xx1.shape, xx1)

	z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T);

	z = z.reshape(xx1.shape);
	plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)
	plt.xlim(xx1.min(), xx1.max())
	plt.ylim(xx2.min(), xx2.max())

	for idx,cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], alpha=0.8, c=cmap(idx), marker=markers[idx], label=cl)
		pass

	plt.xlabel('花瓣长度');
	plt.ylabel('花径长度');
	plt.legend(loc='upper left');
	plt.show();
	pass


# 读取文件
file = './Perceptron/examples.csv';
df = pd.read_csv(file, header=None);
# print(df.head(10))

# 处理第4列表
y = df.loc[0: 100, 4].values;
y = np.where(y == 'Tris-setosa', -1, 1);
# print(y)

# 讲第0和2列取出来分析
X = df.iloc[0: 100, [0, 2]].values;
# print(X)

# 对数据可视化
"""
plt.scatter(X[:5, 0], X[:5, 1], color='red', marker='o', label='setosa'),;
plt.scatter(X[5:10, 0], X[5:10, 1], color='blue', marker='x', label='versicolor');
plt.xlabel('花瓣长度');
plt.ylabel('花径长度');
plt.legend(loc='upper left');
plt.show();
"""

# 训练
ppn = Perceptron(eta=0.1, n_iter=10);
ppn.fit(X, y)

"""
# 训练输出
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o');
plt.xlabel('Epochs');
plt.ylabel('错误分类次数');
plt.show();
"""

# 绘制分割图
plot_decision_regions(X, y, ppn, resolution=0.02)


