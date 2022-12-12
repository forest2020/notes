# Ubuntu下的FlatHub软件仓库

软件仓库安装 https://flatpak.org/setup/Ubuntu

## 安装软件：
```
flatpak install flathub <软件名字（ID）>
```

## 列表已经安装的软件：
```
flatpak list
```

## 运行软件：
```
flatpak run <软件名字（ID）>
```

# linux下视频设备
## 列出视频设备：
```
v4l2-ctl --list-devices
```

## 列出视频设备支持格式：
```
v4l2-ctl --list-formats-ext
v4l2-ctl --list-formats
```

## 使用ffplay打开摄像头预览：
```
ffplay -f v4l2 -video_size 640x480 -i /dev/video0
```

## 使用ffmpeg推摄像头画面到rtmp服务器：
```
ffmpeg -f v4l2 -i /dev/video0 -vcodec h264 -f flv rtmp://192.168.59.129/live/livestream
```

# bash命令
## 暂停前台任务：
```
ctrl + z
```
## 查看暂停的任务：
```
jobs
```
恢复暂停的任务，时期在后台继续运行：
```
bg <jobs列出来的任务号>
```
注意：任务的输出仍然输出到启动时的设备，一般没有指定就是当前控制台。这时看着就很乱了，按ctrl+c也不能结束任务。出现这种情况需要输入“fg <任务号>”，不要关系后台任务输出的内容打乱输入，盲输入就行，然后这个后台任务就转回前台了，就可以用ctrl+c结束了。

## 启动后台任务
```
nohup <任务命令和参数> >xxx.log 2>&1 &
```
启动后台任务，">xxx.log"表示将控制台输出输出到xxx.log中，"2>&1"标识将错误输出到标准输出中，这两个加起来就是把所有的输出记录到xxx.log文件中。

在启动后台的终端查看后台任务：
```
jobs
```
在其他终端查看后台任务：
```
ps -e
```
