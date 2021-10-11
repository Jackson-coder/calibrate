# 相机标定与测距

## 功能说明
本工程文件可以用于双目相机测距工作(版本2，非视差图方法)。

## 使用说明
1.将图片数据放入data文件夹；&nbsp;

2.进入本工程项目的目录；&nbsp;

3.键入以下内容：&nbsp;

```shell
mkdir build
cd build
cmake ..
make
./calibrate
```

Tips: &nbsp;

在执行./calibrate 时使用的是本项目自带的图片数据进行测试，若想使用其他数据，可以通过如下命令查找命令输入方式：

```shell
./calibrate --help
```
