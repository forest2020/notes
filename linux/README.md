linux学习笔记
========
* [Ubuntu下的FlatHub软件仓库](#Ubuntu下的FlatHub软件仓库)
* [linux下视频设备](#linux下视频设备)
* [bash命令](#bash命令)
  * [任务前后台切换](#任务前后台切换)
* [docker命令](#docker命令)
  * [拉取和导入镜像](#拉取和导入镜像)
  * [查看镜像和容器列表](#查看镜像和容器列表)
  * [运行docker](#运行docker)
  * [停止docker](#停止docker)
  * [运行中的docker拷贝文件](#运行中的docker拷贝文件)
  * [运行中的docker操作](#运行中的docker操作)
  * [提交容器修改](#提交容器修改)
  * [保存镜像](#保存镜像)


# Ubuntu下的FlatHub软件仓库

软件仓库安装 https://flatpak.org/setup/Ubuntu

* 安装软件：
```
flatpak install flathub <软件名字（ID）>
```

* 列表已经安装的软件：
```
flatpak list
```

* 运行软件：
```
flatpak run <软件名字（ID）>
```

# linux下视频设备
* 列出视频设备：
```
v4l2-ctl --list-devices
```

* 列出视频设备支持格式：
```
v4l2-ctl --list-formats-ext
v4l2-ctl --list-formats
```

* 使用ffplay打开摄像头预览：
```
ffplay -f v4l2 -video_size 640x480 -i /dev/video0
```

* 使用ffmpeg推摄像头画面到rtmp服务器：
```
ffmpeg -f v4l2 -i /dev/video0 -vcodec h264 -f flv rtmp://192.168.59.129/live/livestream
```

# bash命令
## 任务前后台切换
* 暂停前台任务：
```
ctrl + z
```
* 查看暂停的任务：
```
jobs
```
恢复暂停的任务，时期在后台继续运行：
```
bg <jobs列出来的任务号>
```
注意：任务的输出仍然输出到启动时的设备，一般没有指定就是当前控制台。这时看着就很乱了，按`ctrl+c`也不能结束任务。出现这种情况需要输入`fg <任务号>`，不要关系后台任务输出的内容打乱输入，盲输入就行，然后这个后台任务就转回前台了，就可以用`ctrl+c`结束了。

* 启动后台任务
```
nohup <任务命令和参数> >xxx.log 2>&1 &
```
启动后台任务，`>xxx.log`表示将控制台输出输出到`xxx.log`中，`2>&1`标识将错误输出到标准输出中，这两个加起来就是把所有的输出记录到`xxx.log`文件中。

在启动后台的终端查看后台任务：
```
jobs
```
在其他终端查看后台任务：
```
ps -e
```

# docker命令
## 拉取和导入镜像
* 拉取
从docker的仓库中拉取镜像：
```
docker pull <镜像地址>:<tag>
```
* 导入本地镜像：
```
docker load --input <本地tar镜像文件>
```

## 查看镜像和容器列表
* 查看镜像列表
```
docker images
```
* 查看容器列表
```
docker ps -a
```

## 运行docker
* 从镜像运行，创建新的容器并运行
```
docker run --rm -it -p <主机上的端口>:<docker内的端口> -v <主机上的文件夹>:<docker内的文件夹> <镜像名>:<tag>
```
| 参数 | 说明 |
| ---- | ---- |
| --rm | 推出容器时自动删除该容器，这样启动的容器不会留在`docker ps -a`的列表中 |
| -i | 表示交互式的，表示[cmd]是一个有用户输入的程序，比如/bin/bash和python 等等 |
| -t | 产生一个终端 |
| -p | SOCKET端口映射 |
| -v | 文件夹映射，映射`/etc/localtime`可以同步主机时区和时间 |
| bash | 最后面加上 `bash`，进入docker的控制台，不启动dockers默认的程序，用于调整docker中的文件

* 从容器运行
```
docker start -ia <容器ID>
```
| 参数 | 说明 |
| ---- | ---- |
| -ia | 启动容器，并进入控制台 |

## 停止docker
* 在容器控制台上
```
exit
```
* 在主机上
```
docker stop <容器ID>
```

## 运行中的docker拷贝文件
* 从docker中拷贝文件到主机
```
docker cp <容器ID>:<文件路径> <主机文件夹>
```
* 拷贝主机文件到docker中
```
docker cp <主机文件路径> <容器ID>:<文件夹>
```

## 运行中的docker操作
* 从docker控制台切出到主机
按`ctrl+p + ctrl+q`切出docker，不会停止docker也允许。

* 重新连入docker
```
docker attach <<容器ID>>
```

## 提交容器修改
```
docker commit -a="<作者>" -m="<备注>" <容器ID> <镜像名字>:<tag>
```

## 保存镜像
```
docker save -o <镜像文件的名字(.tar)> <镜像名字>:<tag>
```




















