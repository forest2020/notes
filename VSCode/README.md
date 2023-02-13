调试启动配置文件“launch.json”：
```
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: FastAPI",
            "type": "python",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "--host",
                "192.168.59.129",
                "main:app"
            ],
            "jinja": true,
            "justMyCode": true
        }
    ]
} 
```
调试Python的FastAPI程序，module 中指定调试模块，args 中指定命令行参数。
