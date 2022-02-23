PyTorch学习笔记
========
* [安装](#安装)
* [PyTorch数据类型](#PyTorch数据类型)
 * [数据类型说明](#数据类型说明)
 * [标量](#标量)
* [Visdom可视化](#Visdom可视化)


# 安装
1. 安装Anaconda
2. 打开PyTorch[官网](https://pytorch.org/)
3. 进入Pytorch安装页面，可能是[这个](https://pytorch.org/get-started/locally/)
4. 在“START LOCALLY”标签页里面，根据自己电脑的操作系统和GPU情况，选择好安装条件，复制“Run this Command:”后面的内容
5. 打开Anacoda命令提示符窗口，默认是base虚拟环境（可以创建新的，激活需要的环境），运行上一步复制的命令，完成PyTorch安装

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


创建一个2行3列的符合正态分布的随机2维数组：
```python
import torch

a = torch.randn(2, 3)
print(a)
print(a.type())
print(type(a))
print(isinstance(a, torch.FloatTensor))
print(isinstance(a, torch.cuda.FloatTensor))
print(isinstance(a, torch.DoubleTensor))```
输出：
```
tensor([[-0.1773, -0.4824, -0.1379],
        [ 1.9032, -0.7637, -0.0132]])
torch.FloatTensor
<class 'torch.Tensor'>
True
False
False```
使用```x = x.cuda()```将CPU中的Tensor转入GPU中，前期需要安装支持CUDA的PyTorch。



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
