# -*- coding: utf-8 -*-
import random
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

"""
https://www.zhihu.com/people/codehole/posts?page=6
分类算法和聚类比较类似，都是将输入数据赋予一个标签类别。区别是分类算法的分类是预先确定的，有明确含义的。
而聚类的标签是从输入数据本身的分布中提取出来的一种抽象的类别。聚类是无监督算法，而分类是有监督的，
除了输入数据x外，还有标签y。分类算法非常繁多，朴素贝叶斯分类是其中一种常见的分类算法，它是基于贝叶斯概率推导出来的算法。
该算法在垃圾文本分类中使用非常广泛。sklearn的naive_bayes算法提供了三种实现，BernoulliNB、MultinormialNB和GaussianNB，
BernoulliNB适合抛硬币这种0/1型布尔输入，
MultinormialNB适合文章中的单词数量这种数量型输入，
而最后一个GaussianNB适合本例中连续性数字输入。
接下来我们使用sklearn提供的GaussianNB模块体验一下朴素贝叶斯分类算法。
"""

k = 5
# 预定义的颜色标签
colors = ['green', 'red', 'blue', 'yellow', 'pink']
# 先随机出中心点
centers = []
for i in range(k):
    x = 10 + 100 * random.random()
    y = 10 + 100 * random.random()
    centers.append((x, y))

points = []
# 然后在每个中心点的周围随机100个点
for ci, (x, y) in enumerate(centers):
    ps = []
    for i in range(100):
        px = x + random.random() * 20 - 10
        py = y + random.random() * 20 - 10
        ps.append((px, py))
        points.append(((px, py), ci))
    # 显示数据点
    plt.scatter(
        [x for x, y in ps],
        [y for x, y in ps],
        c=colors[ci], marker='.')

model = GaussianNB()
# 拟合输入
model.fit([p for p, ci in points], [ci for p, ci in points])

pcolors = []
# 网格点
grids = [(x, y) for x in range(0, 120, 5) for y in range(0, 120, 5)]
# 预测颜色
for i, ci in enumerate(model.predict(grids)):
    pcolors.append((grids[i], ci))

# 显示带颜色的网格点
plt.scatter(
    [x for (x, y), ci in pcolors],
    [y for (x, y), ci in pcolors],
    c=[colors[ci] for (x, y), ci in pcolors], marker='x')

plt.show()
