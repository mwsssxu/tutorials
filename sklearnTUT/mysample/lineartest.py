# -*- coding: utf-8 -*-
import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

"""
当我们定义线性回归的损失函数是每个点到直线的距离的平方和时，这种线性回归算法称之为最小二乘法。
回归算法是机器学习的一个基础算法，简单来说就是线性回归，还有非线性回归。
本节我们讲的是最简单的线性回归。线性回归就是用直线来描述两个变量之间的线性关系。
我们在中学时可以根据平面上的两个点来计算出通过这两个点的直线。而线性回归呢跟这个类似，
只不过这里有无穷多个点，我们知道一条直线一般是不能同时通过这无穷多个点的，
所以呢，线性回归要求这条直线像下面的图所显示的那样能大致通过这些点就可以。
而回归的目标就是使得直线尽量不要偏离这些点太远。因为直线要照顾所有的点，
所以要有一个整体性的表达式来衡量直线偏离所有点的程度。然后我们调整直线的系数，
使得这个偏离程度表达式最小化。
"""

xs = range(100)
ys = []
for x in xs:
    ys.append(5 * x + 2 + random.random() * 200)  # 生成随机散点
model = LinearRegression()
model.fit([[x] for x in xs], ys)  # 拟合直线，*转换输入为多维*
ys_ = model.predict([[x] for x in xs])  # 预测所有的样本
plt.scatter(xs, ys, marker='.')  # 画样本点，随机散点
plt.scatter(xs, ys_, marker='+')  # 画预测点，直线点
plt.show()
