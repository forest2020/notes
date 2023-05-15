# Ubuntu 下使用 openssl 创建自签名证书
## 1、安装 openssl
### 1.1、使用 apt 安装
```
sudo apt install openssl
sudo apt install libssl-dev
```
### 1.2、官网下载源码安装（推荐）
#### 1.2.1、官网下载源码压缩包
https://www.openssl.org/source/
#### 1.2.2、解压到本地文件夹
```
tar zxvf openssl-3.1.0.tar.gz
```
#### 1.2.3、进入解压后的文件夹
```
cd openssl-3.1.0
```
#### 1.2.4、设置安装目录
```
./config --prefix=/usr/local/openssl
```
#### 1.2.5、执行命令
```
./config -t
```
#### 1.2.6、安装 gcc
```
sudo apt update
sudo atp install gcc
```
#### 1.2.7、编译
```
make
```
#### 1.2.8、安装
```
make install
```
## 2、生成证书
### 2.1、生成 key 
```
openssl genrsa -des3 -out xxx.key 4096
```
### 2.2、创建自签名证书
```
openssl req -x509 -new -nodes -key XXX.key -sha256 -days 1024 -out XXX.crt
```
### 2.3、将 key 和 crt 转成 pem 格式
```
openssl rsa -in XXX.key -out XXX-key.pem
openssl x509 -in XXX.crt -out XXX-cert.pem
```
