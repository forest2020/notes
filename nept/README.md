# 使用 nodejs + express + prisma + typescript 开发WebApi服务
## 1、安装node.js
参见NodeJS官网（nodejs.org）安装。
## 2、创建WebAPI项目目录
我们使用“myapi”文件夹。    
打开控制台，进入“myapi”文件夹，下面的所有命令都是在此控制台下执行。
## 3、初始化nodejs
使用默认配置初始化。
```
npm init --yes
```
命令执行完成后，目录下产生了package.json文件，内容如下：
```
{
  "name": "myapi",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "keywords": [],
  "author": "",
  "license": "ISC"
}
```
## 4、安装项目依赖的软件包
### 4.1、安装 express 网页应用框架
```
npm install express dotenv 
```
### 4.2、安装 prisma ORM框架
