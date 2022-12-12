# Ubuntu下的FlatHub软件仓库

软件仓库安装 https://flatpak.org/setup/Ubuntu

安装软件：flatpak install flathub <软件名字（ID）>

列表已经安装的软件：flatpak list

运行软件：flatpak run <软件名字（ID）>

# linux下视频设备
列出视频设备：
'''
v4l2-ctl --list-devices
'''

列出视频设备支持格式：
'''
v4l2-ctl --list-formats-ext
v4l2-ctl --list-formats
'''

