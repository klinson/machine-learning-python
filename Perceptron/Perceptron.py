# -*- coding: utf-8 -*-
import numpy as np

class Perceptron(object):
	"""
	Perceptron
	感知器算法
	eta: 学习率
	n_iter: 权重向量的训练次数
	w_: 神经分叉权重向量
	errors_: 用来记录神经元判断出错次数
	"""
	def __init__(self, eta = 0.01, n_iter = 0):
		self.eta = eta;
		self.n_iter = n_iter;
		pass

	def fit(self, X, y):
		"""
		权重更新算法
		根据输入样本，进行神经元培训，x是输入样本向量，y对应样本分类

		X:shape[n_samples, n_features]
		X:[[1, 2, 3], [4, 5, 6]]
		n_samples: 2
		n_features: 3
		"""

		# 初始化权重为0
		# 加一是因为前面算法提到的w0，是步调函数阈值
		self.w_ = np.zeros(1 + X.shape[1]);
		self.errors_ = [];

		for _ in range(self.n_iter):
			errors = 0;
			"""
			X:[[1, 2, 3], [4, 5, 6]]
			y:[1, -1]
			zip(X, y) = [[1, 2, 3, 1], [4, 5, 6, -1]]
			target = 1 / -1
			"""
			for xi, target in zip(X, y):
				"""
				update = n(成功率) * (y - y')，结果为0表示预测正确
				target: y，预定结果
				self.predict(xi): y', 对xi进行预测结果
				"""
				update = self.eta * (target - self.predict(xi));
				

				"""
				xi 是一个向量
				update * xi 等级：
				[▽w(1) = x[1] * update, ▽w(2) = x[2] * update, ▽w(n) = x[n] * update]
				"""
				self.w_[1:] += update * xi;

				# 更新阈值
				self.w_[0] += update;

				errors += int(update != 0.0)
				self.errors_.append(errors);
				pass
			pass
		pass

	def net_input(self, X):
		"""
		实现向量点积
		z = W0*x0+W1*x1+...+Wn*xn;
		"""

		return np.dot(X, self.w_[1:] + self.w_[0])
		pass

	def predict(self, X):
		# 计算xn所属于的分类，先进行点积，再进行判断
		return np.where(self.net_input(X) >= 0.0, 1, -1);
		pass	

