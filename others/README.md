# VMWare nat端口转发
VMWare上运行的虚拟机网卡模式默为nat。VMWare所在主机启用了nat功能，通过nat模式提供其上运行虚拟机的网络访问。   
这种模式下，虚拟机可以自由的访问外面的网络，但是外面的主机无法直接连接到nat后面的到虚拟机，这和家用的路由器一样。为了能在外面访问到虚拟机上提供的服务，需要配置端口转发。  
在Windows上通过网络设置界面找到VMNet8虚拟网卡，配置端口转发。   
在Ubuntu上通过修改主机上的配置文件实现端口转发。    
编辑“/etc/vmware/vmnet8/nat/nat.conf”文件，在“[incomingtcp]”中配置TCP端口转发，在“[incomingudp]”配置UDP端口转发。
```
[incomingtcp]

# Use these with care - anyone can enter into your VM through these...
# The format and example are as follows:
#<external port number> = <VM's IP address>:<VM's port number>
#8080 = 172.16.3.128:80
8000 = 192.168.59.129:8000

[incomingudp]

# UDP port forwarding example
#6000 = 172.16.3.0:6001

```
