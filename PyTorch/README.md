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
* [Visdom可视化](#Visdom可视化)


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
