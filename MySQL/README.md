# 安装数据库
linux下使用apt安装：
```
sudo apt install mysql-server
```
安装成功后，设置root密码：
```
sudo mysql
```
进入mysql控制台后，执行：
```
ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY 'XXX';
```
退出控制台，初始化安全性：
```
mysql_secure_installation
```
输入root密码 XXX，接下来不要用测试密码，第一步选择N。  
接下来选择Y，修改root密码，输入mysql root密码。   
根据需要设置安全性，下面是一个示例：   
不移除默认用户，选择N。  
禁用root远程登录，选择Y。  
移除test数据库，选择Y。  
重新加载特权表，选择Y。  

# 创建数据库
进入mysql控制台：
```
mysql -u root -p
```
创建名字为“mydb”的空白数据库：
```
create database if not exists mydb charset utf8;
```
进入数据库：
```
use mydb;
```
创建名字为“table1”的数据表并增加唯一索引：
```
create table if not exists table1
(
	`ID` int auto_increment primary key COMMENT '自动编号，从1开始，主键',
	`UserName` varchar(128) not null COMMENT '用户名，用户唯一标识',
	`Disabled` tinyint DEFAULT 0 not null COMMENT '用户停用，1表示停用，0表示可用，默认0',
	`Memo` varchar(128) COMMENT '备注'
);

ALTER TABLE `User` ADD UNIQUE (`UserName`);
```
# 使用数据库
_在mysql控制台中执行下面语句。_   
显示数据库列表：
```
show databases;
```
打开数据库：
```
use mydb;
```
显示数据表列表：
```
show tables;
```
显示表结构字段信息：
```
desc table1;
```
显示表的创建脚本：
```
show create table table1;
```
删除索引：
```
alter table table1 drop index UserName;
```
删除数据库：
```
drop database mydb;
```
表添加字段：
```
alter table table1 add column Password varchar(64) not null comment '用户密码' after UserName;
```
存储过程中的变量声明，必须在开头的“begin”后面声明，前面不能有其它语句，不能再中间声明：
```
create procedure maintaindb()
begin
    declare UserID,RoleID int default 0;
    select ID into UserID from User where UserName='admin';
    select ID into RoleID from Role where RoleName='系统管理员';    
end
```
在sql脚本中创建存储过程，需要先将语句分隔符从默认的“;”改为其它符号，创建结束后在改回来：
```
-- abc.sql

-- 修改语句分隔符
delimiter //

create procedure maintaindb()
begin
  ...
end //

-- 恢复语句分隔符
delimiter ;

```
