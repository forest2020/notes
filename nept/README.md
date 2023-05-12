# 使用 nodejs + express + prisma + typescript 开发 WebApi 服务
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
### 4.2、安装 prisma ORM 框架
```
npm install prisma
```
### 4.3、安装用于开发的 typescirpt 环境
```
npm i -D typescript @types/express @types/node
```
安装成功后，package.json 文件中增加了 "devDependencies" 部分，如下：
```
{
  "name": "yjaigcapi",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "test": "echo \"Error: no test specified\" && exit 1"
  },
  "keywords": [],
  "author": "",
  "license": "ISC",
  "dependencies": {
    "dotenv": "^16.0.3",
    "express": "^4.18.2",
    "prisma": "^4.14.0"
  },
  "devDependencies": {
    "@types/express": "^4.17.17",
    "@types/node": "^20.1.3",
    "typescript": "^5.0.4"
  }
}

```
"devDependencies" 用于开发阶段，产品发布后不需要。
### 4.4、安装用于调试的包
允许同时执行多个命令的支持包 concurrently    
监控源码修改，自动重启服务器包 nodemon
```
npm install -D concurrently nodemon
```
## 5、初始化TypeScript
```
npx tsc --init
```
初始化成功后，文件夹下产生了 tsconfig.json 文件，内容如下：
```
{
  "compilerOptions": {
     "target": "es2016",
     "module": "commonjs",
     "esModuleInterop": true,
     "forceConsistentCasingInFileNames": true,
     "strict": true,
     "skipLibCheck": true
  }
}
```
将 "outDir" 取消注释，打开，设置为 ./disk, 设置后配置文件如下：、
```
{
  "compilerOptions": {
     "target": "es2016",
     "module": "commonjs",
     "outDir": "./dist",
     "esModuleInterop": true,
     "forceConsistentCasingInFileNames": true,
     "strict": true,
     "skipLibCheck": true
  }
}
```
## 6、在myapi文件夹下创建 index.ts
index.ts 内容如下：
```
import express, { Express, Request, Response } from 'express';
import dotenv from 'dotenv';

dotenv.config();

const app: Express = express();
const port = process.env.PORT;

app.get('/', (req: Request, res: Response) => {
  res.send('Express + TypeScript Server');
});

app.listen(port, () => {
  console.log(`⚡️[server]: Server is running at http://localhost:${port}`);
});
```
myapi目录下新建 .nev 文件，内容如下：
```
# express web service port
PORT=8000
```
## 7、编译并以开发模式启动Web服务
### 7.1、打开 package.json，修改 "scripts" 中的内容，修改后内容如下：
```
{
  "name": "myapi",
  "version": "1.0.0",
  "description": "",
  "main": "index.js",
  "scripts": {
    "build": "npx tsc",
    "start": "node dist/index.js",
    "dev": "concurrently \"npx tsc --watch\" \"nodemon -q dist/index.js\""
  },
  "keywords": [],
  "author": "",
  "license": "ISC",
  "dependencies": {
    "dotenv": "^16.0.3",
    "express": "^4.18.2",
    "prisma": "^4.14.0"
  },
  "devDependencies": {
    "@types/express": "^4.17.17",
    "@types/node": "^20.1.3",
    "concurrently": "^8.0.1",
    "nodemon": "^2.0.22",
    "typescript": "^5.0.4"
  }
}
```
### 7.2、编译程序
```
npm run build
```
### 7.3、以开发模式启动程序
```
npm run dev
```
启动成功后，在浏览器中打开 http://localhost:8000/
可能到如下内容代表 WebApi服务搭建成功：
```
Express + TypeScript Server
```
