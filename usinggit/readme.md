1、本地电脑安装git，配置git  
```
git config --global user.name "John Doe"   
git config --global user.email johndoe@example.com   
git config --global core.editor vim
```

2、打开GitBash，进入项目目录，初始git  
```
git init  
```

3、编辑项目根目录下 .gitignore，设置忽略的文件  
  
4、将所有文件加入到git缓存  
```
git add .   
```

5、提交   
```
git commint  
```  
在弹出的编辑器中输入提交的说明  

6、在GitHub上创建新的库，根据说明设置本地项目的remoate，并把本地库推到GitHub上。  
  
7、本地新建分支，并推送到git  
```
git branch iss53 
```

8、本地提交后，如果不需要合并到主分支，只想把新分支推到GitHub上 git push origin iss53  
