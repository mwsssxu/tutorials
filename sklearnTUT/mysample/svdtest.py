# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.decomposition import TruncatedSVD

"""
打个比方说一张女人图片，我们如何判定这个女人是不是美女呢。我们会看比较关键的一些特征，比如说脸好不好看，胸好不好看，屁股怎么样，腿怎么样，至于衣服上是某个花纹还是手臂上有一个小痔还是，这些特征我们都是不关心的，就可以过滤掉。我们关心的是主成分，也就是对结果贡献系数较大的特征。SVD算法的作用就是来告诉你哪些特征是重要的，有多重要，哪些特征是不重要的，是可以忽略的。

接下来我们使用sklearn提供的TruncatedSVD模块来对美女图片进行压缩。

首先我们使用matplotlib显示一张美女png图片，png图片的格式非常简单，每一个像素有三个维度的颜色值RGB，整个图片就是一个「height x width x 3」维的矩阵。

"""

# 加载png数据矩阵
img_array = img.imread('test2.png')

shape = img_array.shape
print(shape)
# 高度、宽度、RGB通道数=3
height, width, channels = shape[0], shape[1], shape[2]

# 转换成numpy array
img_matrix = np.array(img_array)

# 存储RGB三个通道转换后的数据
planes = []

# RGB三个通道分别处理
for idx in range(channels):
    # 提取通道
    plane = img_matrix[:, :, idx]
    # 转成二维矩阵
    plane = np.reshape(plane, (height, width))
    # 保留10个主成分
    svd = TruncatedSVD(n_components=10)
    # 拟合数据，进行矩阵分解，生成特征空间，剔去无关紧要的成分
    svd.fit(plane)
    # 将输入数据转换到特征空间
    new_plane = svd.transform(plane)
    # 再将特征空间的数据转换会数据空间
    plane = svd.inverse_transform(new_plane)
    # 存起来
    planes.append(plane)

# 合并三个通道平面数据
img_matrix = np.dstack(planes)
# 显示处理后的图像
plt.imshow(img_matrix)

plt.show()
