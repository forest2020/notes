# 安装
1. 安装Anaconda
2. 打开PyTorch[官网](https://pytorch.org/)
3. 进入Pytorch安装页面，可能是[这个](https://pytorch.org/get-started/locally/)
4. 在“START LOCALLY”标签页里面，根据自己电脑的操作系统和GPU情况，选择好安装条件，复制“Run this Command:”后面的内容
5. 打开Anacoda命令提示符窗口，默认是base虚拟环境（可以创建新的，激活需要的环境），运行上一步复制的命令，完成PyTorch安装

# Visdom可视化
安装：pip install visdom
启动服务：
```
python -m visdom.server
```
启动成功后显示查看图标的网站，一般是 http://http://localhost:8097/
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
