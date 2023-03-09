# 列出已经安装的包
查看当前项目已安装包（项目跟目录必须有 package.json 文件），默认列出第一层，使用 ```-depth 数字```控制列出的层数：
```
npm ls
```
只显示生产环境依赖的包:
```
npm ls --omit=dev
```
只显示开发环境依赖的包
```
npm ls --include=dev
```
查看全局已安装（-g 的意思是 global 全局的意思）：
```
npm ls -g
```
查找包是否安装：
```
npm ls | grep 包含的关键字
```
