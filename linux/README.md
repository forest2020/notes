linux学习笔记
========
* [Ubuntu下的FlatHub软件仓库](#Ubuntu下的FlatHub软件仓库)
* [linux下视频设备](#linux下视频设备)
* [软磁盘阵列管理](#软磁盘阵列管理)
* [重置损坏的磁盘分区表](#重置损坏的磁盘分区表)
* [bash命令](#bash命令)
  * [任务前后台切换](#任务前后台切换)
  * [查看程序运行所需的共享库](#查看程序运行所需的共享库)
  * [命令参数中使用命令](#命令参数中使用命令)
  * [获取IP地址字符串](#获取IP地址字符串)
  * [主机间复制文件](#主机间复制文件)
  * [进程操作](#进程操作)
  * [tail命令显示日志](#tail命令显示日志)
  * [查看文件夹占用的空间](#查看文件夹占用的空间)
* [docker命令](#docker命令)
  * [拉取和导入镜像](#拉取和导入镜像)
  * [查看镜像和容器列表](#查看镜像和容器列表)
  * [运行docker](#运行docker)
  * [停止docker](#停止docker)
  * [删除容器](#删除容器)
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

# 软磁盘阵列管理

mdadm 是 多磁盘和设备管理(Multiple Disk and Device Administration) 的缩写。它是一个命令行工具，可用于管理 Linux 上的软件 RAID 阵列。本文概述了使用它的基础知识。

以下 5 个命令是你使用 mdadm 的基础功能：

* 创建 RAID 阵列：```mdadm --create /dev/md/test --homehost=any --metadata=1.0 --level=1 --raid-devices=2 /dev/sda1 /dev/sdb1```
* 组合（并启动）RAID 阵列：```mdadm --assemble /dev/md/test /dev/sda1 /dev/sdb1```
* 停止 RAID 阵列：```mdadm --stop /dev/md/test```
* 删除 RAID 阵列：```mdadm --zero-superblock /dev/sda1 /dev/sdb1```
* 检查所有已组合的 RAID 阵列的状态：```mdadm -D /dev/md/test 或者 cat /proc/mdstat```

功能说明
mdadm –create

上面的创建命令除了 -create 参数自身和设备名之外，还包括了四个参数：

1、–homehost：

默认情况下，mdadm 将你的计算机名保存为 RAID 阵列的属性。如果你的计算机名与存储的名称不匹配，则阵列将不会自动组合。此功能在共享硬盘的服务器群集中很有用，因为如果多个服务器同时尝试访问同一驱动器，通常会发生文件系统损坏。名称 any 是保留字段，并禁用 -homehost 限制。

2、 –metadata：

-mdadm 保留每个 RAID 设备的一小部分空间，以存储有关 RAID 阵列本身的信息。 -metadata 参数指定信息的格式和位置。1.0 表示使用版本 1 格式，并将元数据存储在设备的末尾。

3、–level：

-level 参数指定数据应如何在底层设备之间分布。级别 1 表示每个设备应包含所有数据的完整副本。此级别也称为磁盘镜像。

4、–raid-devices：

-raid-devices 参数指定将用于创建 RAID 阵列的设备数。

通过将 -level=1（镜像）与 -metadata=1.0 （将元数据存储在设备末尾）结合使用，可以创建一个 RAID1 阵列，如果不通过 mdadm 驱动访问，那么它的底层设备会正常显示。这在灾难恢复的情况下很有用，因为即使新系统不支持 mdadm 阵列，你也可以访问该设备。如果程序需要在 mdadm 可用之前以只读访问底层设备时也很有用。例如，计算机中的 UEFI 固件可能需要在启动 mdadm 之前从 ESP 读取引导加载程序。


mdadm –assemble

如果成员设备丢失或损坏，上面的组合命令将会失败。要强制 RAID 阵列在其中一个成员丢失时进行组合并启动，请使用以下命令：

 ```mdadm --assemble --run /dev/md/test /dev/sda1```

其他重要说明

避免直接写入底层是 RAID1 的设备。这导致设备不同步，并且 mdadm 不会知道它们不同步。如果你访问了在其他地方被修改了设备的某个 RAID1 阵列，则可能导致文件系统损坏。如果你在其他地方修改 RAID1 设备并需要强制阵列重新同步，请从要覆盖的设备中删除 mdadm 元数据，然后将其重新添加到阵列，如下所示：

```
mdadm --zero-superblock /dev/sdb1
mdadm --assemble --run /dev/md/test /dev/sda1
mdadm /dev/md/test --add /dev/sdb1
```

以上用 sda1 的内容完全覆盖 sdb1 的内容。

要指定在计算机启动时自动激活的 RAID 阵列，请创建 /etc/mdadm.conf 配置。


# 重置损坏的磁盘分区表
如果一个磁盘在磁盘管理工具中无法添加、删除分区，或者无法正确查看分区，那么需要重置磁盘的分布表，使用parted命令：   
1、进入parted交互环境：```sudo parted /dev/<磁盘ID>```   
2、重置分区表：```(parted)    mklabel   gpt```   
3、在磁盘管理工具（界面工具，如Windows的“磁盘管理”、Ubuntu的“磁盘”）中创建新的分区。  

**注意：重置分区表后磁盘上的数据全部丢失**



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

## 查看程序运行所需的共享库
```
ldd <可执行文件>
```
列出程序所需要的库，如果行的后面出现`not found`表示缺少这个库。

## 命令参数中使用命令
* 方法1
```
命令 <反向的撇>命令<反向的撇>
```
反向撇中的命令输出作为前面命令的参数

* 方法2
```
命令 $(命令)
```

## 获取IP地址字符串
```
ifconfig <网卡名> | grep 'inet ' | awk '{print $2}'
```

## 主机间复制文件
linux与linux、linux与windows之间复制文件。  
从远程主机拷贝文件到本机：
```
scp -r -p <用户>@<IP>:<源路径> <目标路径>
```
从本机拷贝文件到远程主机：
```
scp -r -p <源路径> <用户>@<IP>:<目标路径>
```
| 参数 | 说明 |
| ---- | ---- |
| -r | 复制目录及所有子目录和文件，递归方式 |
| -p | 复制文件的修改时间 |

## 进程操作
* 列出系统中当前正在运行的那些进程。
使用ps 命令，支持的 3 种语法格式：  
    1. UNIX 风格。         选项可以组合在一起，并且选项前必须有 "-" 连字符  
    2. BSD  风格。         选项可以组合在一起，但是选项前不能有 "-" 连字符  
    3. GNU  风格的长选项。 选项前有 两个 "-" 连字符  

命令用法：
ps [options] [--help]

常用命令参数：

ps命令常用用法（方便查看系统进程）  
ps c 列出程序时，显示每个程序真正的指令名称，而不包含路径，参数或常驻服务的标示。  
ps -e 此参数的效果和指定"A"参数相同。  
ps e 列出程序时，显示每个程序所使用的环境变量。  
ps f 用ASCII字符显示树状结构，表达程序间的相互关系。  
ps -H 显示树状结构，表示程序间的相互关系。  
ps -N 显示所有的程序，除了执行ps指令终端机下的程序之外。  
ps s 采用程序信号的格式显示程序状况。  
ps S 列出程序时，包括已中断的子程序资料。  
ps -t <终端机编号> 　指定终端机编号，并列出属于该终端机的程序的状况。  
ps u 　以用户为主的格式来显示程序状况。  
ps x 　显示所有程序，不以终端机来区分。  

常用组合:  
ps a 显示现行终端机下的所有程序，包括其他用户的程序。  
ps -A 显示所有进程。  
ps aux 显示所有包含其他使用者的进程  
ps -e 显示所有进程。  
ps e 显示所有进程，包括环境变量。  

* 像进程发送信号
使用 kill 和 killall 命令。  
```
kill -<信号> <进程ID>
killall -<信号> <进程名>
```
常用信号:  
| 信号编号 |	信号名 |	含义 |
| ---- | ---- | ---- |
| 0 |	EXIT | 程序退出时收到该信息。 |
| 1 |	HUP |	挂掉电话线或终端连接的挂起信号，这个信号也会造成某些进程在没有终止的情况下重新初始化。 |
| 2 |	INT |	表示结束进程，但并不是强制性的，常用的 "Ctrl+C" 组合键发出就是一个 kill -2 的信号。 |
| 3 |	QUIT |	退出。 |
| 9 |	KILL |	杀死进程，即强制结束进程。 |
| 11 |	SEGV |	段错误。 |
| 15 |	TERM |	正常结束进程，是 kill 命令的默认信号。 |

## tail命令显示日志
```
tail -n 10 -f abc.log
```
实时刷新显示abc.log的后10行内容

## 查看文件夹占用的空间
查看当前目录总共占的容量。而不单独列出各子项占用的容量
```
du -sh
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
| bash | 最后面加上 `bash`，进入docker的控制台，不启动dockers默认的程序，用于调整docker中的文件 |
| --privileged | 使用docker的超级权限 |
| --evn | 添加docker控制台环境变量 |

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

## 删除容器
在主机上：
```
docker rm <容器ID>
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
docker attach <容器ID>
```

* 再开一个终端
```
docker exec -it <容器ID> bash
```

## 提交容器修改
```
docker commit -a="<作者>" -m="<备注>" <容器ID> <镜像名字>:<tag>
```

## 保存镜像
```
docker save -o <镜像文件的名字(.tar)> <镜像名字>:<tag>
```




















