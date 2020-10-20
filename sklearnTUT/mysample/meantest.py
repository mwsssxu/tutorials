# -*- coding: utf-8 -*-
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
"""
https://zhuanlan.zhihu.com/p/34547528
聚类是无监督算法，只提供了输入数据x，而没有参考目标y。聚类的目标就是将输入数据进行分类，距离接近的放到一个分类，距离远的就分开。
那如何用数学语言来衡量聚类的目标呢？ 聚类算法中最简单也是最常见的算法就是Kmeans算法。Kmeans算法将聚类的目标定义为寻找
最佳的K个中心点。就好比你要在K个城市开肯德基，你应该选择每个城市中心地点来开设，这样距离人群的平均距离最近。
一旦有了K个中心点，那么对于任意输入数据，我们计算它和所有中心点之间的距离，取最近的中心点作为其类别即可。
Kmeans算法必须由人工指定聚类的数量K，然后算法会自动去寻找最佳的K个中心点，并将输入数据点归类。至于如何确定K，
真实的数据往往是不能像上图那样肉眼就可以观察出有几个分类。真实的数据往往是多维数据结构，难以使用图形来直接呈现的。
Kmeans算法必须由人工指定初始中心，然后算法使用迭代的方式来移动这K个中心点，最终收敛到最佳中心点位置。
这个最佳中心点是和初始点的位置相关的，也就是说初始的选择会影响到最终的中心点的结果。
确定初始中心也有相应的算法，不过不在本文的讨论范围。
那K到底该如何确定呢？这个也不在本文的讨论范围，小编后续再讲。这里就假设我们已经知道了K是多少，
然后使用Kmeans算法来把输入数据点划分为最佳的K个堆。
接下来，我们使用sklearn提供的KMeans模块来实践一下。KMeans模块默认提供了确定初始中心点的算法，
用户可以不必关心中心点初始化的问题，但是用户必须指定K值。
"""
k = 5
centers = []
# 先随机出K个中心点
for i in range(k):
    x = 100 * random.random()
    y = 100 * random.random()
    centers.append((x, y))

# 对每个中心点周围再随机出N个点
points = []
for x, y in centers:
    for i in range(100):
        px = x + random.random() * 20 - 10
        py = y + random.random() * 20 - 10
        points.append((px, py))

plt.scatter([px for px, py in points], [py for px, py in points], marker='.')
plt.scatter([x for x, y in centers], [y for x, y in centers], marker='+')

# Kmeans模型只需要设置聚类数量，初始中心点已经有默认算法提供了
model = KMeans(n_clusters=5)
model.fit(points)  # 拟合数据点
centers_ = model.cluster_centers_
# 显示聚类中心
plt.scatter([x for x, y in centers_], [y for x, y in centers_], marker='^')

clusters = []
for i in range(k):
    clusters.append([])
# 预测所有网格点的类别，不同类别的网格点放在不同的数组中
grids = [(x, y) for x in range(-10, 110, 5) for y in range(-10, 110, 5)]
for i, n in enumerate(model.predict(grids)):
    clusters[n].append(grids[i])

# 显示所有类别的网格点
for i in range(k):
    plt.scatter([x for x, y in clusters[i]], [y for x, y in clusters[i]], marker='x')

plt.show()
