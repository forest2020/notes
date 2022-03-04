PyTorch学习笔记
========
* [安装](#安装)
* [IDE](#IDE)
* [PyTorch数据类型](#PyTorch数据类型)
  * [数据类型说明](#数据类型说明)
  * [标量](#标量)
  * [向量](#向量)
  * [张量初始化](#张量初始化)
* [索引与切片](#索引与切片)
* [维度变换](#维度变换)
* [广播](#广播)
* [拼接与分割](#拼接与分割)
* [数学运算](#数学运算)
* [统计](#统计)
* [比较运算](#比较运算)
* [激活函数](#激活函数)
* [损失和梯度计算](#损失和梯度计算)
* [优化器](#优化器)
* [Visdom可视化](#Visdom可视化)
* [搭建一个简单的分类网络](#搭建一个简单的分类网络)
* [正则化Regularization](#正则化Regularization)
  * [范数](#范数)
  * [深度学习中的正则化](#深度学习中的正则化)
* [学习率](#学习率)
* [早停](#早停)
* [丢弃](#丢弃)


# 安装
1. 安装Anaconda
2. 打开PyTorch[官网](https://pytorch.org/)
3. 进入Pytorch安装页面，可能是[这个](https://pytorch.org/get-started/locally/)
4. 在“START LOCALLY”标签页里面，根据自己电脑的操作系统和GPU情况，选择好安装条件，复制“Run this Command:”后面的内容
5. 打开Anacoda命令提示符窗口，默认是base虚拟环境（可以创建新的，激活需要的环境），运行上一步复制的命令，完成PyTorch安装

# IDE
* Spyder，Anaconda自带的IDE，启动速度比较满，支持调试，智能提示不好用。
* VSCode，速度快，智能提示好用。

在Anaconda虚拟环境下使用VSCode方法：打开Anaconda命令提示符窗口，zixing```code.exe```启动VSCode，需要python插件，在插件的设置中指定Anacoda安装路径。



# PyTorch数据类型
## 数据类型说明
PyTorch中存储数据的容器叫做张量（Tensor），支持整数和浮点数，不支持字符串，与Python中数据类型对应关系为：
| Python | PyTorch |
| ---- | ---- |
| int | IntTensor of size() |
| float | FloatTensor of size() |
| int array | IntTensor of size [d1,d2,...] |
| float array | FloatTensor of size [d1,d2,...] |
| string | -- |

PyTorch中的张量可以在CPU中也可以在GPU中，使用to device完成。
PyTorch是一个数值计算库，不是完备的编程语言库，因此不需要支持字符串操作，如果需要字符串，可以用编码表示。

## 标量
维度是0的张量，声名方法：
```python
import torch

a = torch.tensor(1)
b = torch.tensor(2.)
print(a, a.type(), a.shape)
print(b, b.type(), b.shape)
```
输出：
```
tensor(1) torch.LongTensor torch.Size([])
tensor(2.) torch.FloatTensor torch.Size([])
```
size是空的方括号（[]）代表标量，在深度学习中，训练中的损失loss就是一个标量。

## 向量
维度大于0的张量，可以理解为数组，支持任意维度。

创建一个2行3列的符合正态分布的随机2维数组：
```python
import torch

a = torch.randn(2, 3)
print(a)
print(a.type())
print(type(a))
print(isinstance(a, torch.FloatTensor))
print(isinstance(a, torch.cuda.FloatTensor))
print(isinstance(a, torch.DoubleTensor))
```
输出：
```
tensor([[-0.1773, -0.4824, -0.1379],
        [ 1.9032, -0.7637, -0.0132]])
torch.FloatTensor
<class 'torch.Tensor'>
True
False
False
```
使用```x = x.cuda()```将CPU中的Tensor转入GPU中，前期需要安装支持CUDA的PyTorch。

1度向量的创建：
```python
import torch

a = torch.tensor([7])
b = torch.tensor([2.])
print(a, a.type(), a.shape)
print(b, b.type(), b.shape)
```
输出：
```
tensor([7]) torch.LongTensor torch.Size([1])
tensor([2.]) torch.FloatTensor torch.Size([1])
```
从numpy数组创建向量：
```python
import torch
import numpy as np

data = np.ones(2)
a = torch.from_numpy(data)
print(data, type(data), data.shape)
print(a, a.type(), a.shape)
# shape转换为python列表
print(list(a.shape))
```
输出：
```
[1. 1.] <class 'numpy.ndarray'> (2,)
tensor([1., 1.], dtype=torch.float64) torch.DoubleTensor torch.Size([2])
[2]
```

## 张量初始化
```python
import torch

# 为初始化的张量，具有内存中原有的数据，最好不要使用这种方法初始化
a = torch.empty(2, 5)
print('a:', a)

# 随机初始化张量
# 初始化[0,1]之间的均匀分布的张量
b = torch.rand(2, 5)
print('b:', b)

# 初始化标准正太分布的张量
c = torch.randn(2, 5)
print('c:', c)

# 创建一个与已有张量同样形状的张量
d = torch.rand_like(a)
print('d:', d)

# 创建一个整数张量，指定最小值和最大值整数，包括最小值，不包括最大值，放回取样
# 最后一个参数是维度
e = torch.randint(3, 6, [2, 4])
print('e:', e)

# 创建一个数列，第三个参数可以指定间隔（步长）
f = torch.arange(2, 10).reshape(2, 4)
print('f:', f)
# 在开始和结束区间内创建第三个参数指定个数的数列，包括两端点
f1 = torch.linspace(2, 10, 3)
print('f1:', f1)

# 创建一个全部是同一个数值的张量
g = torch.full([3, 2], 7)
print('g:', g)

# 创建全是1的张量
h = torch.ones(3, 2)
print('h:', h)

# 创建全是0的张量
i = torch.ones(3, 2)
print('i:', i)

# 创建单位矩阵
j = torch.eye(3)
print('j:', j)
j1 = torch.eye(3, 4)
print('j1', j1)

# 创建一个随机打散的索引张量
k = torch.randperm(10)
print('k:', k)
```
输出：
```
a: tensor([[0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.]])
b: tensor([[0.6758, 0.9819, 0.7428, 0.5941, 0.6671],
        [0.5848, 0.2866, 0.6752, 0.1474, 0.6199]])
c: tensor([[-0.9216,  0.7161, -0.6093,  0.5157,  0.9894],
        [-1.7610, -0.9583, -1.2302, -0.5157,  0.8143]])
d: tensor([[0.0291, 0.7307, 0.4028, 0.6687, 0.5023],
        [0.4898, 0.4763, 0.1095, 0.9657, 0.8446]])
e: tensor([[3, 3, 5, 3],
        [5, 5, 4, 5]])
f: tensor([[2, 3, 4, 5],
        [6, 7, 8, 9]])
f1: tensor([ 2.,  6., 10.])
g: tensor([[7, 7],
        [7, 7],
        [7, 7]])
h: tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])
i: tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])
j: tensor([[1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]])
j1 tensor([[1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.]])
k: tensor([9, 3, 4, 7, 0, 6, 5, 2, 1, 8])
```


# 索引与切片
```python
import torch

# 创建一个4维向量，一般是图像识别网络的输入（batch_size, channel, width, height)
a = torch.rand(4, 3, 28, 28)

# 维度的索引从0开始，小于维度的数目，从左面开始
# 取第0个维度的第0项目，是一个3维的向量
print('a[0] shape:', a[0].shape)

# []内用逗号间隔维度
# 取维度0的第1个项目， 是一个2维的向量
print('a[1,2] shape:', a[1, 2].shape)
# 指定所有维度，取到的是一个标量
print('a[2,1,3,5) shape:', a[2, 1, 3, 5].shape)

# 使用冒号指定范围，冒号前面是开始位置，后面是结束位置，不知道默认开始或者结尾
# 取第0维度的前2项，其它维度全部取，取到的是4维数据，第0维的数量是2
print('a[:2] shape:', a[:2].shape)
# 取第0维度中间2项，第3维度的第10项后面的数据，得到（2，3，18，28）的向量
print('a[1:3,:,10:] shape:', a[1:3, :, 10:].shape)

# 第二个冒号表示步长
# 取第0维度中间2项，第3维度的第10项后面间隔1项取1个，得到（2，3，9，28）的向量
print('a[1:3,:,10::2] shape:', a[1:3, :, 10::2].shape)

# 使用索引数组选择项
# 取第三个通道的3、5项，得到(4,3,28,2)
print('a.index_select(3,[3,5]) shape:',
      a.index_select(3, torch.tensor([3, 5])).shape)

# 取所有维度所有数据
print('a[...] shape:', a[...].shape)
# 最后一个维度取前2项，得到（4，3，28，2）
print('a[...,:2] shape:', a[..., :2].shape)

# 使用掩码向量选择数据
# 得到大于1的项的掩码向量,mask是一个（4，3，28，28）的向量，a中大于0.5的项目对应mask位置是True，其它地方是False
mask = a > 0.5
print('mask part data:', mask[0, 0, 0])
# 通过掩码选择后的数据拉直为一个1维向量，长度等于a中大于0.5项的个数
print('a中大于0.5的项:', a.masked_select(mask))

# 拉直成1维向量根据位置取项
print('a拉直后的2、3项:', a.take(torch.tensor([2, 3])))
```
输出：
```
a[0] shape: torch.Size([3, 28, 28])
a[1,2] shape: torch.Size([28, 28])
a[2,1,3,5) shape: torch.Size([])
a[:2] shape: torch.Size([2, 3, 28, 28])
a[1:3,:,10:] shape: torch.Size([2, 3, 18, 28])
a[1:3,:,10::2] shape: torch.Size([2, 3, 9, 28])
a.index_select(3,[3,5]) shape: torch.Size([4, 3, 28, 2])
a[...] shape: torch.Size([4, 3, 28, 28])
a[...,:2] shape: torch.Size([4, 3, 28, 2])
mask part data: tensor([ True, False, False,  True,  True, False, False,  True, False, False,
         True,  True,  True,  True, False,  True, False, False, False,  True,
        False,  True, False, False,  True, False, False, False])
a中大于0.5的项: tensor([0.9761, 0.9434, 0.8226,  ..., 0.7555, 0.6995, 0.6488])
a拉直后的2、3项: tensor([0.4858, 0.9434])
```

# 维度变换
| 操作 | 说明 |
| --- | --- |
| View/reshape | 保持项目总数量和顺序不变的情况下，改变维度 |
| Squeeze/unsqueeze | 挤压长度是1的维度/增加长度为1的维度 |
| Transpose/t/permute | 矩阵的转置 |
| Expand/repeat | 维度的扩展 |

view和reshape完全一样，为了与numpy一致，使用reshape更习惯一些。
```python
import torch

a = torch.rand(4, 1, 28, 28)
print('a shape:', a.shape)

# reshape
a = a.reshape(4, 28, 28)
print('reshape:', a.shape)

# unsqueeze
# 在第1个维度前面插入一个维度，得到(4,1,28,28)
a = a.unsqueeze(1)
print('unsqueeze(1):', a.shape)
# 在最后面插入一个维度，得到(4,1,28,28,1)
a = a.unsqueeze(-1)
print('unsqueeze(-1):', a.shape)

# squeeze，指定一个要删减的维度，对长度不是1维度无效，不指定删除全部长度是1的维度
# 删减第0个维度，第0个维度长度是4，操作不做任何改变，得到(4,1,28,28,1)
a = a.squeeze(0)
print('squeeze(0):', a.shape)
# 删减所有长度是1的维度，得到(4,28,28)
a = a.squeeze()
print('squeeze():', a.shape)

# expand，广播模式，不会复制数据，效率高，节省内存
a = a.unsqueeze(1)
# 将a的第1个维度（一般代表图片的色彩通道数，1是灰度，3是彩色）扩展到3
# 被扩展的维度的长度原来必须是1，原来是（4，1，28，28），扩展后是（4，3，28，28）
a = a.expand(-1, 3, -1, -1)
print('a.expend(-1,3,-1,-1):', a.shape)

# repeat，拷贝数据，参数指定每个维度拷贝的次数
# 第0个维度拷贝2次，第1个维度拷贝1次，第2、3维度拷贝10次，得到（8，3，280，280）
a = a.repeat(2, 1, 10, 10)
print('a.repeat(2,1,10,10):', a.shape)

# t()/T，矩阵的转置，只能用于2维矩阵
# transpose，交换维度，可以用于任何维度的向量
# 交换0和1维度，得到（3，8，280，280）,
# 交换后内存的数据不连续，需要使用contiguous()让数据连续，然后才可以做reshape操作，否则reshape得到是交换前数据顺序
a = a.transpose(0, 1)
print('a.transpose(0,1):', a.shape)
```
输出：
```
a shape: torch.Size([4, 1, 28, 28])
reshape: torch.Size([4, 28, 28])
unsqueeze(1): torch.Size([4, 1, 28, 28])
unsqueeze(-1): torch.Size([4, 1, 28, 28, 1])
squeeze(0): torch.Size([4, 1, 28, 28, 1])
squeeze(): torch.Size([4, 28, 28])
a.expend(-1,3,-1,-1): torch.Size([4, 3, 28, 28])
a.repeat(2,1,10,10): torch.Size([8, 3, 280, 280])
a.transpose(0,1): torch.Size([3, 8, 280, 280])
```

# 广播
```python
import torch

a = torch.tensor([[1, 2], [3, 4], [5, 6]])
print('a shape:', a.shape)
print('a data:', a)

b = torch.tensor([3])
print('b shape:', b.shape)
print('b data:', b)

# 广播是从最小维度匹配
# 计算a+b时，b通过广播扩展为（3,2)维度，每个元素都是3
print('a+b:', a+b)

# 匹配最后一个维度，长度必须等于a最后一个维度的长度，每个元素分别操作
c = torch.tensor([2, 1])
print('a+c:', a+c)

# a的维度是[4,32,28,30],b的维度是[1,32,1,1]
# a+b也是可以广播的，32不变，第一个1变为4，第二个1变为28，第三个1变为30
# a+b得到的维度是[4,32,28,30]
```
输出：
```
a shape: torch.Size([3, 2])
a data: tensor([[1, 2],
        [3, 4],
        [5, 6]])
b shape: torch.Size([1])
b data: tensor([3])
a+b: tensor([[4, 5],
        [6, 7],
        [8, 9]])
a+c: tensor([[3, 3],
        [5, 5],
        [7, 7]])
```

# 拼接与分割
| 操作 | 说明 |
| --- | --- |
| cat | 连接多个向量的某一维度，其它维度必须一样 |
| stack | 叠加多个向量的某一维度，被叠加向量的所有维度必须一样，得到叠加向量在指定叠加维度前面增加了一个维度，长度等于被叠加向量的个数 |
| split | 在指定维度上拆分，指定返回的每个向量拆分维度的长度，返回一个元组，维度不变 |
| chunk | 指定返回向量拆分维度的长度进行拆分，返回一个元组，数量根据每个的数量计算，最后一个小于等于指定长度 |

```python
import torch

# cat,从指定的维度上连接，不指定默认为0维度
# 在默认0维度上链接，其它维度必须一致
a = torch.rand(5, 12, 7)
b = torch.rand(10, 12, 7)
c = torch.cat([a, b])
print('a b cat at dim 0 shape:', c.shape)

# 在1维度上连接
a = torch.rand(5, 12, 7)
b = torch.rand(5, 2, 7)
c = torch.cat([a, b], dim=1)
print('a b cat at dim 1 shape:', c.shape)

# stack，相加的维度前面增加一个新的维度
# 叠加的向量维度必须一样
a = torch.rand(5, 8, 7)
b = torch.rand(5, 8, 7)
c = torch.rand(5, 8, 7)
d = torch.stack([a, b, c], dim=1)
print('a b c stack at dim 1 shape:', d.shape)

# split，指定长度拆分
a = torch.rand(5, 3, 2)
b, c = a.split([2, 3])
print('b shape:', b.shape)
print('c shape:', c.shape)

# chunk，指定个数拆分
d = a.chunk(3)
print('d len:', len(d))
print('d[0] shape:', d[0].shape)
print('d[1] shape:', d[1].shape)
print('d[2] shape:', d[2].shape)
```
输出：
```
a b cat at dim 0 shape: torch.Size([15, 12, 7])
a b cat at dim 1 shape: torch.Size([5, 14, 7])
a b c stack at dim 1 shape: torch.Size([5, 3, 8, 7])
b shape: torch.Size([2, 3, 2])
c shape: torch.Size([3, 3, 2])
d len: 3
d[0] shape: torch.Size([2, 3, 2])
d[1] shape: torch.Size([2, 3, 2])
d[2] shape: torch.Size([1, 3, 2])
```

# 数学运算
标量预算没有特别之处，主要需要了解是向量的运算
```python
import torch

# +、-、*、/，对应元素加、减、乘、除，支持广播
a = torch.rand(3, 4)
print('a shape:', a.shape)
b = torch.rand(4)
print('b shape:', b.shape)
print('a+b shape:', (a+b).shape)

# 矩阵乘法，mm只支持2维矩阵乘法，@/matmul支持多维矩阵
# 多维矩阵乘法对最后2个维度做矩阵乘法，前面的维度支持广播，一一对应
a = torch.rand(4, 3, 28, 64)
print('a shape:', a.shape)
b = torch.rand(4, 1, 64, 28)
print('b shape:', b.shape)
print('a@b shape:', (a@b).shape)

# **n/pow(n) n次方，sqrt开方，rsqrt平凡跟倒数
a = torch.tensor([[2, 3], [4, 5]])
print('a:', a)
print('a的平方：', a**2)
print('a的3次方：', a.pow(3))
print('a的平方根：', a.sqrt())
print('a的平方根倒数：', a.rsqrt())

# log,对数，e为底的对数，其它低速的函数：log10,log2
print('[e,e]的log:', torch.tensor([2.7183, 2.7183]).log())

# floor向下取整,ceil向上取整,trunc取整数部分,frac取小数部分,round四舍五入
a = torch.tensor(3.14)
print('a: ', a)
print('floor       ceil        trunc         frac')
print(a.floor(), a.ceil(), a.trunc(), a.frac())

# clamp，裁剪，限制数据的范围
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print('a:', a)
print('a.clamp(4):', a.clamp(4))
print('a.clamp(3,5):', a.clamp(3, 5))
```
输出：
```
a shape: torch.Size([3, 4])
b shape: torch.Size([4])
a+b shape: torch.Size([3, 4])
a shape: torch.Size([4, 3, 28, 64])
b shape: torch.Size([4, 1, 64, 28])
a@b shape: torch.Size([4, 3, 28, 28])
a: tensor([[2, 3],
        [4, 5]])
a的平方： tensor([[ 4,  9],
        [16, 25]])
a的3次方： tensor([[  8,  27],
        [ 64, 125]])
a的平方根： tensor([[1.4142, 1.7321],
        [2.0000, 2.2361]])
a的平方根倒数： tensor([[0.7071, 0.5774],
        [0.5000, 0.4472]])
[e,e]的log: tensor([1.0000, 1.0000])
a:  tensor(3.1400)
floor       ceil        trunc         frac
tensor(3.) tensor(4.) tensor(3.) tensor(0.1400)
a: tensor([[1, 2, 3],
        [4, 5, 6]])
a.clamp(4): tensor([[4, 4, 4],
        [4, 5, 6]])
a.clamp(3,5): tensor([[3, 3, 3],
        [4, 5, 5]])
```


# 统计
| 操作 | 说明 |
| --- | --- |
| norm | 求二范数 |
| mean | 求均值 |
| sum | 求和 |
| prod | 累乘 |
| max、min、argmax、argmin | 最大、最小值，最大、最小值的索引 |
| kthvalue、topk | 第几个的值、前几个的值 |

```python

import torch

a = torch.full([8], 1.)
b = a.view(2, 4)
c = a.view(2, 2, 2)
print('a:', a, a.shape)
print('b:', b, b.shape)
print('c:', c, c.shape)

# norm
# 1范数，所有元素的和
print('a.norm(1):', a.norm(1))
print('b.norm(1):', b.norm(1))
print('c.norm(1):', c.norm(1))
# 2范数，所有元素的平方和再开方
print('a.norm(2):', a.norm(2))
print('b.norm(2):', b.norm(2))
print('c.norm(2):', c.norm(2))
# 指定维度的范数,维度的理解不太清楚，大约是取那个维度就消掉那个维度
print('b.norm(1,dim=1):', b.norm(1, dim=1))
print('c.norm(1,dim=1):', c.norm(1, dim=1))
print('c.norm(1,dim=0):', c.norm(1, dim=0))

# mean、sum、max、min、prod，argmax、argmin
a = torch.arange(1, 9).view(2, 4).float()
print('a:', a)
print('a.mean():', a.mean())
print('a.sum():', a.sum())
print('a.max():', a.max())
print('a.min():', a.min())
print('a.prod(),累乘:', a.prod())
print('a.argmax():', a.argmax())
print('a.argmin(1):', a.argmin(1))
print('a.argmin(1,keepdim=True):', a.argmin(1, keepdim=True))
print('a.topk(2),largest=Flase返回最小的k个:', a.topk(2))
print('a.kthvalue(2),第k小的:', a.kthvalue(2))
```
输出：
```
a: tensor([1., 1., 1., 1., 1., 1., 1., 1.]) torch.Size([8])
b: tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.]]) torch.Size([2, 4])
c: tensor([[[1., 1.],
         [1., 1.]],

        [[1., 1.],
         [1., 1.]]]) torch.Size([2, 2, 2])
a.norm(1): tensor(8.)
b.norm(1): tensor(8.)
c.norm(1): tensor(8.)
a.norm(2): tensor(2.8284)
b.norm(2): tensor(2.8284)
c.norm(2): tensor(2.8284)
b.norm(1,dim=1): tensor([4., 4.])
c.norm(1,dim=1): tensor([[2., 2.],
        [2., 2.]])
c.norm(1,dim=0): tensor([[2., 2.],
        [2., 2.]])
a: tensor([[1., 2., 3., 4.],
        [5., 6., 7., 8.]])
a.mean(): tensor(4.5000)
a.sum(): tensor(36.)
a.max(): tensor(8.)
a.min(): tensor(1.)
a.prod(),累乘: tensor(40320.)
a.argmax(): tensor(7)
a.argmin(1): tensor([0, 0])
a.argmin(1,keepdim=True): tensor([[0],
        [0]])
a.topk(2),largest=Flase返回最小的k个: torch.return_types.topk(
values=tensor([[4., 3.],
        [8., 7.]]),
indices=tensor([[3, 2],
        [3, 2]]))
a.kthvalue(2),第k小的: torch.return_types.kthvalue(
values=tensor([2., 6.]),
indices=tensor([1, 1]))
```

# 比较运算
```python
import torch

#  >、>=、<、<=、!=、==，返回与原向量同形的True、False对比结果向量
a = torch.arange(1, 9).view(2, 4).float()
print('a:', a)
print('a>2:', a > 2)

# where 条件筛选
cond = torch.tensor([[1, 1, 2, 2], [2, 2, 1, 1]])
print('条件cond：', cond)
b = torch.ones(8).reshape(2, 4)
print('b:', b)
print('a.where(cond<2,b):', a.where(cond < 2, b))

# gather,查表操作，将原表的一个维度的数值作为字典表的索引，替换成字典表的值
a = torch.tensor([[1, 0, 1], [0, 0, 0]])
b = torch.tensor([8, 9, 10])
print('torch.gather(b.expand(2,3),dim=1,index=a):',
      torch.gather(b.expand(2, 3), dim=1, index=a))
```
输出：
```
a: tensor([[1., 2., 3., 4.],
        [5., 6., 7., 8.]])
a>2: tensor([[False, False,  True,  True],
        [ True,  True,  True,  True]])
条件cond： tensor([[1, 1, 2, 2],
        [2, 2, 1, 1]])
b: tensor([[1., 1., 1., 1.],
        [1., 1., 1., 1.]])
a.where(cond<2,b): tensor([[1., 2., 1., 1.],
        [1., 1., 7., 8.]])
torch.gather(b.expand(2,3),dim=1,index=a): tensor([[9, 8, 9],
        [8, 8, 8]])
```


# 激活函数
```python
import torch

a = torch.linspace(-100, 100, 10)
print('a:', a)

# sigmoid，可以看到当x较大或者较小时的值都是1和0，梯度消失了
print('sigmoid(a):', torch.sigmoid(a))

# ReLU，有效的防止了梯度的消失
print('relu(a):', torch.relu(a))
```
输出：
```
a: tensor([-100.0000,  -77.7778,  -55.5556,  -33.3333,  -11.1111,   11.1111,
          33.3333,   55.5556,   77.7778,  100.0000])
sigmoid(a): tensor([0.0000e+00, 1.6655e-34, 7.4564e-25, 3.3382e-15, 1.4945e-05, 9.9999e-01,
        1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00])
relu(a): tensor([  0.0000,   0.0000,   0.0000,   0.0000,   0.0000,  11.1111,  33.3333,
         55.5556,  77.7778, 100.0000])
```


# 损失和梯度计算
```python
import torch
import torch.nn as nn
from torch.nn import functional as F

# 均方差损失，MSE
x = torch.tensor(2.)
print('x:', x)
# 需要求导的张量必须将requires_grad设置为True
w = torch.full([1], 3., requires_grad=True)
print('w:', w)
mse = F.mse_loss(torch.tensor([5.7]), x*w)
print('mse loss(2*3-5.7)**2=0.09:', mse)

# 损失对w的梯度，导函数在x点的值
print('d(loss)/d(w)=2*(x*w-5.7)*x=2*(2*3-5.7)*2:',
      torch.autograd.grad(mse, [w]))

# 第二种方法计算梯度，backward方法
# 上面读取了一次梯度后没有设置保留，不能第二次读取，需要再次计算
mse = F.mse_loss(torch.tensor([5.7]), x*w)
mse.backward()
print('mse.backward(), w.grad:', w.grad)

# 用于分类的softmax，网络的输出通过softmax转换成概率
logits = torch.rand(3, requires_grad=True)
print('logits:', logits)
pred = F.softmax(logits, dim=0)
print('pred:', pred)
print('pred sum:', pred.sum())
# pred.backward()会报错，backward只能对标量进行
# 查看logits的导数值，注意必须设置保留计算图，否则无法获取后面2个元素的导数值
print('d(pred[0])/logits grads:',
      torch.autograd.grad(pred[0], [logits], retain_graph=True))
print('d(pred[1])/logits grads:',
      torch.autograd.grad(pred[1], [logits], retain_graph=True))
print('d(pred[2])/logits grads:',
      torch.autograd.grad(pred[2], [logits], retain_graph=True))
# 定义真是的标签
labels = torch.tensor([1., 0, 0])
print('labels:', labels)
# 定义交叉熵对象
criteon = nn.CrossEntropyLoss()
# 计算交叉熵损失,需要2维的向量，第0维是批次，第1维是预测概率和实际标签
loss = criteon(pred.unsqueeze(0), labels.unsqueeze(0))
print('loss:', loss)
# 后向传播
loss.backward()
# 显示梯度
print('d(loss)/logits grads:', logits.grad)
```
输出：
```
x: tensor(2.)
w: tensor([3.], requires_grad=True)
mse loss(2*3-5.7)**2=0.09: tensor(0.0900, grad_fn=<MseLossBackward0>)
d(loss)/d(w)=2*(x*w-5.7)*x=2*(2*3-5.7)*2: (tensor([1.2000]),)
mse.backward(), w.grad: tensor([1.2000])
logits: tensor([0.4118, 0.6768, 0.7699], requires_grad=True)
pred: tensor([0.2678, 0.3491, 0.3831], grad_fn=<SoftmaxBackward0>)
pred sum: tensor(1.0000, grad_fn=<SumBackward0>)
d(pred[0])/logits grads: (tensor([ 0.1961, -0.0935, -0.1026]),)
d(pred[1])/logits grads: (tensor([-0.0935,  0.2272, -0.1337]),)
d(pred[2])/logits grads: (tensor([-0.1026, -0.1337,  0.2363]),)
labels: tensor([1., 0., 0.])
loss: tensor(1.1653, grad_fn=<DivBackward1>)
d(loss)/logits grads: tensor([-0.2025,  0.0944,  0.1081])
```


# 优化器
求Himmelblau二元函数的最小值的x,y

![alt Himmelblau Function](./images/himmelblau.jpg)

函数的最小值点：

![alt Himmelblau Minima](./images/himmelblau_minima.jpg)

使用matplotlib绘制函数曲面：
```python
import numpy as np
from matplotlib import pyplot as plt


def himmelblau(x):
    # 定义himmelblau函数，传入一个1位数组，有2个元素，x[0]代表x，x[1]代表y
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2


# 生成x、y±6之间的网格数据，间隔0.1
x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
X, Y = np.meshgrid(x, y)
print('X,Y maps:', X.shape, Y.shape)

# 计算函数值,传入了两个二维向量，对应元素计算函数值，返回一个[120,120]的向量
Z = himmelblau([X, Y])
print('Z shape:', Z.shape)

# 显示函数的曲面
fig = plt.figure('himmelblau')
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z)
ax.view_init(30, -30)  # 沿着Z轴旋转30°，沿着Y轴旋转-30
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()
```
输出：
```
X,Y maps: (120, 120) (120, 120)
Z shape: (120, 120)
```
画出来的函数曲面图：

![alt himmelblau_surface](./images/himmelblau_surface.jpg)

使用梯度下降方法求函数的最小值：
```python
import torch


def himmelblau(x):
    # 定义himmelblau函数，传入一个1位数组，有2个元素，x[0]代表x，x[1]代表y
    return (x[0]**2+x[1]-11)**2+(x[0]+x[1]**2-7)**2


# 从（0,0）点开始沿着梯度下降方向寻找函数的最小值点
x = torch.tensor([0., 0.], requires_grad=True)
# 定义一个优化器
optimizer = torch.optim.Adam([x], lr=1e-3)  # 优化对象是x向量，学习率是0.001
for step in range(20000):
    # 计算当前点的函数值
    z = himmelblau(x)
    
    # 优化器的用法：梯度先清零，函数值反向传播，优化器调整下一步的X值
    optimizer.zero_grad()
    z.backward()
    optimizer.step()

    if step % 2000 == 0:
        print('step {}: x={}, f(x)={}'.format(step, x.tolist(), z.item()))
```
输出：
```
step 0: x=[0.0009999999310821295, 0.0009999999310821295], f(x)=170.0
step 2000: x=[2.3331806659698486, 1.9540694952011108], f(x)=13.730916023254395
step 4000: x=[2.9820079803466797, 2.0270984172821045], f(x)=0.014858869835734367
step 6000: x=[2.999983549118042, 2.0000221729278564], f(x)=1.1074007488787174e-08
step 8000: x=[2.9999938011169434, 2.0000083446502686], f(x)=1.5572823031106964e-09
step 10000: x=[2.999997854232788, 2.000002861022949], f(x)=1.8189894035458565e-10
step 12000: x=[2.9999992847442627, 2.0000009536743164], f(x)=1.6370904631912708e-11
step 14000: x=[2.999999761581421, 2.000000238418579], f(x)=1.8189894035458565e-12
step 16000: x=[3.0, 2.0], f(x)=0.0
step 18000: x=[3.0, 2.0], f(x)=0.0
```
找到了函数的一个最小值点(3,2)，x初始化不同的位置，可以找到临近的另外3个最小值点。

动量（momentum）：torch.optim.SDG的momentum参数，取值范围0~1，代表取上一次梯度向量大小的比例，用乘以这个比例的上一次梯度向量与本次的梯度向量相加，得到本次更新的最终向量，这样可以避免陷入局部最小值出不来，造成网络训练效果差。torch.optim.Adam函数中没有momentum参数，Adam算法内置了动量的支持。


# Visdom可视化
安装：pip install visdom
启动服务：
```
python -m visdom.server
```
启动成功后显示查看图标的网站，一般是 `http://localhost:8097/`
如果能打开这个网页，表示启动成功了。
在python程序中想visdom服务添加数据，实时显示图表。

在网页中添加一个图标：
```python
from visdom import Visdom
viz=Visdom()
viz.line([[0.,0.]],[0.],win='train1',opts={'title':'train loss&acc','legend':['loss','acc.']})
```
上面代码在visdom网页中初始化了一个窗口，显示了训练的loss和acc.，第一个参数提供一个数组，指定Y的第一个值，第二个参数指定X的第一个值，第三个参数是窗口的名字，第四个参数是图标的名字，第五个参数对应Y数据的标题

实时添加数据：
```python
viz.line([[loss.item(),acc.item()]],[global_step],win='train1', update='append')
```


# 搭建一个简单的分类网络
```pthin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from visdom import Visdom

# 定义批次、学习率、训练次数
batch_size = 200
learning_rate = 1e-3
epochs = 10

# 使用手写数字数据集，做一个0~9手写数字的识别
# 加载训练集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
# 加载测试集
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)

# 定义网络，一个4层的全连接网络
# 第1层784个神经元，等于图像的像素个数，灰度图像,28*28，共782个像素
w1, b1 = torch.randn(300, 784, requires_grad=True), torch.randn(
    300, requires_grad=True)
# 第2层300个神经元，因此第1层输出就是300
w2, b2 = torch.randn(200, 300, requires_grad=True), torch.randn(
    200, requires_grad=True)
# 第3层200个神经元，因此第2层输出就是200
w3, b3 = torch.randn(100, 200, requires_grad=True), torch.randn(
    100, requires_grad=True)
# 第4层100个神经元，因此第3层输出就是100，共10个标签（0~9），输出为标签数量
w4, b4 = torch.randn(10, 100, requires_grad=True), torch.randn(
    10, requires_grad=True)

# 用正态分布初始化参数，这样才能在训练中找到全局最小值，否则找到的是局部最小值
torch.nn.init.kaiming_normal(w1)
torch.nn.init.kaiming_normal(w2)
torch.nn.init.kaiming_normal(w3)
torch.nn.init.kaiming_normal(w4)


def forward(x):
    # 前向传播函数
    x = x@w1.T+b1
    x = F.relu(x)

    x = x@w2.T+b2
    x = F.relu(x)

    x = x@w3.T+b3
    x = F.relu(x)

    x = x@w4.T+b4
    x = F.relu(x)

    return x


# 优化器，使用Adam
optimizer = optim.Adam([w1, b1, w2, b2, w3, b3], lr=learning_rate)

# 损失函数，交叉熵
criteon = nn.CrossEntropyLoss()

# 可视化训练过程，使用Visdom，观察训练集的损失和准去率
viz = Visdom()
viz.line([[0., 0.]], [0.], win='train1', opts={
         'title': 'train loss&acc', 'legend': ['loss', 'acc.']})

# 训练
global_step = 0
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # 彩色通道是1，去掉它
        data = data.view(-1, 28*28)

        # 得到网络的推断的特征向量
        logits = forward(data)
        # 这一步做了sofmat和交叉熵损失
        loss = criteon(logits, target)

        # 反向传播，调整参数w和b
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每训练10次显示一组数据到Visdom中
        if batch_idx % 10 == 0:
            pred = logits.argmax(1)
            acc = pred.eq(target).sum()/len(target)
            viz.line([[loss.item(), acc.item()]], [global_step],
                     win='train1', update='append')
            global_step += 1
            viz.images(data.view(-1, 1, 28, 28)[0:10], win='x')
            viz.text(str(pred.detach().numpy()),
                     win='pred', opts={'title': 'pred'})

        # 每100次打印一次训练结果
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    # 训练结束，测试模型的性能
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28*28)
        logits = forward(data)

        # 累加测试集上的损失
        test_loss += criteon(logits, target)

        # 从特征向量数组中过滤出每个向量最大值的索引，这个索引对应就是数字
        pred = logits.data.max(1)[1]
        # 与标签对比，累加正确的识别的图像数量
        correct += pred.eq(target.data).sum()

    #
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
```
输出：
```
....
Test set: Average loss: 0.0015, Accuracy: 8905/10000 (89%)

Train Epoch: 9 [0/60000 (0%)]	Loss: 0.255767
Train Epoch: 9 [20000/60000 (33%)]	Loss: 0.268368
Train Epoch: 9 [40000/60000 (67%)]	Loss: 0.171398

Test set: Average loss: 0.0014, Accuracy: 8932/10000 (89%)
```
训练过程中损失和准确率变化曲线：

![alt MNIST_train_chart](./images/MNIST_train_chart.jpg)




# 正则化Regularization
## 范数
范数是衡量某个向量空间（或矩阵）中的每个向量以长度或大小。范数的一般化定义：对实数p>=1， 范数定义如下：

![alt norm formula](./images/norm_formula.png)

* L1范数：当p=1时，是L1范数，其表示某个向量中所有元素绝对值的和。
* L2范数：当p=2时，是L2范数， 表示某个向量中所有元素平方和再开根， 也就是欧几里得距离公式。

## 深度学习中的正则化
* L1正则化（Lasso回归）算法：

![alt regularization L1](./images/regularization_L1.png)
* L2正则化（岭回归）算法：

![alt regularization L2](./images/regularization_L2.png)

* 弹性网回归算法：

![alt ElasticNet](./images/regularization_ElasticNet.png)

PyTorch中没有L1正则化和弹性网回归的实现，L2正则化示例如下：
```python
import torch
import torch.nn as nn

# 构建一个1层的神经网络，使用2个神经元，输入3个数字，输出2个数字
# 每次输入2组数据
labels = torch.tensor([[0, 1.], [1, 0]])
ce = nn.CrossEntropyLoss()

# 不使用L2正则化
print('----- not L2 regularization -----')
w1 = torch.tensor([[1, 1.2, 1.3], [1.1, 0.9, 1.4]], requires_grad=True)
x1 = torch.tensor([[12., 11., 10.], [15., 14., 13.]])

optimizer1 = torch.optim.SGD([w1], lr=0.1)

logits1 = x1@w1.T
print('logits1=x1@w1.T:', logits1)

loss1 = ce(logits1, labels)
print('ce loss:', loss1)
w1_grad = torch.autograd.grad(loss1, w1, retain_graph=True)
print('d(loss1)/d(w1):', w1_grad)
print('manual cal new w1, w1-lr*w1_grad:', w1-0.1*w1_grad[0])

optimizer1.zero_grad()
loss1.backward()
optimizer1.step()
print('backward new w1:', w1)

# 使用L2正则化
print('----- L2 regularization -----')
w2 = torch.tensor([[1, 1.2, 1.3], [1.1, 0.9, 1.4]], requires_grad=True)
x2 = torch.tensor([[12., 11., 10.], [15., 14., 13.]])

optimizer2 = torch.optim.SGD([w2], lr=0.1, weight_decay=0.1)

logits2 = x2@w2.T
print('logits2=x2@w2.T:', logits2)

loss2 = ce(logits2, labels)
print('ce loss:', loss2)
w2_grad = torch.autograd.grad(loss2, w2, retain_graph=True)
print('d(loss2)/d(w2):', w2_grad)
print('manual cal new w2, w2-lr*w2_grad-lr*lambda*w2:',
      w2-0.1*w2_grad[0]-0.1*0.1*w2)

optimizer2.zero_grad()
loss2.backward()
optimizer2.step()
print('backward new w2:', w2)
```
输出
```
----- not L2 regularization -----
logits1=x1@w1.T: tensor([[38.2000, 37.1000],
        [48.7000, 47.3000]], grad_fn=<MmBackward0>)
ce loss: tensor(0.8039, grad_fn=<DivBackward1>)
d(loss1)/d(w1): (tensor([[ 3.0179,  2.7417,  2.4655],
        [-3.0179, -2.7417, -2.4655]]),)
manual cal new w1, w1-lr*w1_grad: tensor([[0.6982, 0.9258, 1.0534],
        [1.4018, 1.1742, 1.6466]], grad_fn=<SubBackward0>)
backward new w1: tensor([[0.6982, 0.9258, 1.0534],
        [1.4018, 1.1742, 1.6466]], requires_grad=True)
----- L2 regularization -----
logits2=x2@w2.T: tensor([[38.2000, 37.1000],
        [48.7000, 47.3000]], grad_fn=<MmBackward0>)
ce loss: tensor(0.8039, grad_fn=<DivBackward1>)
d(loss2)/d(w2): (tensor([[ 3.0179,  2.7417,  2.4655],
        [-3.0179, -2.7417, -2.4655]]),)
manual cal new w2, w2-lr*w2_grad-lr*lambda*w2: tensor([[0.6882, 0.9138, 1.0404],
        [1.3908, 1.1652, 1.6326]], grad_fn=<SubBackward0>)
backward new w2: tensor([[0.6882, 0.9138, 1.0404],
        [1.3908, 1.1652, 1.6326]], requires_grad=True)
```


# 学习率
开始训练的时候学习率设置大一些，比如0.01、0.001，随着训练的进行，当快到达最低点的时候，应该减小学习率，这样才能更准确的到达最低点，否则因为学习率大，每次W调整的步长大，会在最低点左右摆动，而始终无法到达最低点。学习率的调整方案可以采用条件式或者固定式两种方案。
* 条件式学习率调整方案

```python
# 优化器，初始学习率是0.1
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# 学习率优化方案，监控的指标在patience个step没有变化，下一个step时就将lr缩小factor倍
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

#训练
for epoch in range(10):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)
```
* 固定式学习率调整方案
```python
# 优化器，初始学习率是0.1
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
# 学习率优化方案，每30个step，lr缩小为原来的gamma倍
scheduler=torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

#训练
for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```


# 早停
早停(Early Stopping),训练过程中保存验证集性能最好的模型参数，当在指定的epoch后验证集性能不在提升，停止训练。早停可以防止过拟合，同时可以尽早的结束训练，避免无意义的时间和算力的浪费。注意，验证集的性能曲线未必是单调增的，早停的设置需要经验尝试，避免局部性能曲线下降错误的停止训练，导致无法得到最佳性能的模型。

# 丢弃
丢弃（Dropout），随机丢掉一些神经元，相当于将神经元的输出置为0，这样可以防止模型过拟合。
```python
import torch
import torch.nn as nn
import numpy as np

# 定义一个随机丢弃20%输入数据的Dropout对象，输入向量的元素有20%会被置为0
m = nn.Dropout(p=0.2)

# 定义一个输入向量，作为Dropout的输入，做1000次随机丢弃实验实验
input = torch.randn(20, 16)
percentage_retain = []
for i in range(1000):
    output = m(input)
    # 将每次保留下来的百分比保存到列表，不等于0的元素个数/元素总数
    percentage_retain.append((output != 0).sum()/input.numel())


# 验证1000次保留下来的百分比平均值和理论上的0.8（1-0.2）是否一致
print('average percentage of 1000 times dropout：', np.array(
    percentage_retain).sum() / len(percentage_retain))
```
输出：
```
average percentage of 1000 times dropout： 0.80015625
```
实验证明nn.Dropout中设置的概率对应丢弃输入的百分比。

使用PyTorch搭建一个有Dropout的网络：
```python
import torch.nn as nn

# 定义一个有Dropout的网络
net = nn.Sequential(
    nn.Linear(784, 300),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(300, 100),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(100, 10)
)

for epoch in range(1000):
    # 在训练的时候，需要做Dropout，切换成训练模型
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        ...

    # 在评估中，需要关闭Dropout，切换成评估模式
    net.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        ...
```







