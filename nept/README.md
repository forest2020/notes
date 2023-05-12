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
  "name": "myapi",
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
只需要在项目第一次运行前编译一次，否则第一次运行 dev 命令会有一个异常。
### 7.3、以开发模式启动程序
```
npm run dev
```
启动成功后，在浏览器中打开 http://localhost:8000/
可能到如下内容代表 WebApi服务搭建成功：
```
Express + TypeScript Server
```
## 8、初始化 prisma ORM
```
npx prisma init
```
成功后，在 myapi 文件夹下新产生了 prisma 文件夹，并且 .evn文件中增加了 DATABASE_URL 项目，内容如下：
```
# express web service port
PORT=8000

# This was inserted by `prisma init`:
# Environment variables declared in this file are automatically made available to Prisma.
# See the documentation for more detail: https://pris.ly/d/prisma-schema#accessing-environment-variables-from-the-schema

# Prisma supports the native connection string format for PostgreSQL, MySQL, SQLite, SQL Server, MongoDB and CockroachDB.
# See the documentation for all the connection string options: https://pris.ly/d/connection-strings

DATABASE_URL="postgresql://johndoe:randompassword@localhost:5432/mydb?schema=public"
```
## 9、配置 prisma
我们这里使用mysql数据库，打开 prisma/schema.prisma，将数据库修改为 mysql，修改后内容如下：
```
// This is your Prisma schema file,
// learn more about it in the docs: https://pris.ly/d/prisma-schema

generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "mysql"
  url      = env("DATABASE_URL")
}

```
打开 .env 文件，配置 mysql 连接字符串，修改后的内容如下：
```
# express web service port
PORT=8000

# This was inserted by `prisma init`:
# Environment variables declared in this file are automatically made available to Prisma.
# See the documentation for more detail: https://pris.ly/d/prisma-schema#accessing-environment-variables-from-the-schema

# Prisma supports the native connection string format for PostgreSQL, MySQL, SQLite, SQL Server, MongoDB and CockroachDB.
# See the documentation for all the connection string options: https://pris.ly/d/connection-strings

DATABASE_URL="mysql://root:123456@localhost:3306/mydb"
```
## 10、编写数据库脚本
打开 prisma/schema.prisma，在最下面增加数据库脚本，增加后的内容如下：
```
// This is your Prisma schema file,
// learn more about it in the docs: https://pris.ly/d/prisma-schema

generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "mysql"
  url      = env("DATABASE_URL")
}

model Role {
  id            Int      @id @default(autoincrement())
  name          String
  description   String?
  users         UserRole[]
}

model Group {
  id     Int     @id @default(autoincrement())
  name          String
  description   String?
  groups        UserGroup[]
}

model UserRole {
  user       User   @relation(fields: [userId], references: [id])
  userId     Int
  Role       Role   @relation(fields: [roleId], references: [id])
  roleId     Int

  @@id([userId, roleId])
}

model UserGroup {
  user       User   @relation(fields: [userId], references: [id])
  userId     Int
  group      Group   @relation(fields: [groupId], references: [id])
  groupId    Int

  @@id([userId, groupId])
}

model User {
  id         Int      @id @default(autoincrement())
  account    String   @unique
  password   String?
  name       String?
  createdAt  DateTime @default(now())
  updatedAt  DateTime @updatedAt
  roles      UserRole[]
  groups     UserGroup[]
}
```
## 11、创建数据库
```
npx prisma migrate dev --name init
```
查看 mysql 的 mydb 数据库，应该有 user、role、group、usergroup和userrole表。
## 12、测试 prisma 
手工在 user 表中增加一条记录。    
打开 index.ts，增加查询数据库代码，增加好的代码如下：
```
import express, { Express, Request, Response } from 'express';
import dotenv from 'dotenv';
import { PrismaClient, User } from '@prisma/client'

dotenv.config();

const app: Express = express();
const port = process.env.PORT;

const prisma = new PrismaClient()

app.get('/', async (req: Request, res: Response) => {
    let allUsers: User[] = [];
    let err = '';
    await prisma.user.findMany({ include: { roles: true, groups: true } })
        .then((users) => allUsers = users)
        .catch((e) => err = e)
        .finally(async () => await prisma.$disconnect());
    res.send(`Express + TypeScript Server. Users: [${allUsers.map((user) => user.name)}]`);
});

app.listen(port, () => {
    console.log(`⚡️[server]: Server is running at http://localhost:${port}`);
});
```
打开浏览器，访问 http://localhost:8000/，显示如下内容：
```
Express + TypeScript Server. Users: [用户1]
```
## 13、使用 VS Code 调试ts代码
### 13.1、配置 ts 和 js 代码之间的映射
打开 tsconfig.json 文件，打开 ts 与 js 代码之间的映射选项 sourceMap，修改后的文件内容是：
```
{
  "compilerOptions": {
     "target": "es2016",
     "module": "commonjs",
     "sourceMap": true,
     "outDir": "./dist",
     "esModuleInterop": true,
     "forceConsistentCasingInFileNames": true,
     "strict": true,
     "skipLibCheck": true
  }
}
```
### 13.2、启动调试
在 VS Code 中打开 myapp 文件夹，打开 package.json文件，点击 "scripts"行上面的 Debug，选择 dev。     
在 ts 文件增加断点，调试。
