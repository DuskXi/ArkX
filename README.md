# 明日方舟长草期清理智神器

### 这是一个经过深思熟虑和大量实验后准备开写的项目

 
---

# 这玩意能干啥？

---

这个项目主要是为了各位博士在鹰角长草期时不必过于痛苦和伤肝而存在(当然重点在于不用花钱去tb找代肝)

现阶段目标是作为针对单个关卡的连续指定次数循环，针对于某些理智添加剂仓鼠玩家来说很不错，尤其是爆刷活动时

或者单日挂机，也就是每天只需要上线一次挂起来，保证一滴理智也不会浪费

至于什么基建自动换班和指定不同关卡之类的大饼，慢慢来好了，等项目先写出来再说

毕竟都在长草期了，大家估计也不在乎每天只对基建进行一次换班带来的损失了，理智能tm清掉已经不错了

# 安装&使用

---

***该教程需要在Powershell中运行，请不要使用cmd，如果你使用的是conda，只允许执行pip命令时使用cmd***

1. 创建一个目录，在该目录下执行: `git clone https://github.com/DuskXi/ArkX.git`
2. 执行 `cd ArkX`
3. 此处分两种情况，一种是懒得自己配置python环境的，直接去Release中下载配置好的python环境（过阵子就会发布），如果自己配置python环境，请跟随(4)开始的步骤
4. 安装好python环境以及配置好环境变量, 3.7.x 以上即可
5. 执行以下命令:

```bash
pip install -r requirements.txt
pip install lxml matplotlib pathlib PaddlePaddle
mkdir apiInstallTemp
cd apiInstallTemp
```

6. 然后前往 *https://github.com/protocolbuffers/protobuf/releases* 下载最新版本的win-64的zip,
   然后把压缩包中的bin目录中的protoc.exe复制到ArkX/apiInstallTemp目录下
7. 继续执行以下命令:
8. 拉取指定版本TensorFlow model

```bash
git clone -b v1.13.0 https://github.com/tensorflow/models.git
```

10. 切换目录，执行protoc:

```bash
cd models/research
../../protoc.exe object_detection/protos/*.proto --python_out=.
Get-ChildItem object_detection/protos/*.proto | foreach {../../protoc.exe "object_detection/protos/$($_.Name)" --python_out=.}
```

11. 安装object_detection_api:

```bash
pip install .
```

12. 安装pycocotools:

```bash
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```
13. 修改部分import, 在目录`Lib\site-packages\object_detection\utils`中修改以下文件中的`import tensorflow as tf` 为 `import tensorflow.compat.v1 as tf`
```
label_map_util.py
ops.py
visualization_utils.py
```

15. 从这步开始，自行安装环境和直接下载环境的开始合并
16. 安装完成后，在ArkX目录下执行以下命令(如果是直接从Release中下载的python环境，则需要在python前面加上路径，或者自行设置环境变量):

```bash
python main.py
```

# 功能进度

---

## 已经实现的功能(核心功能)

1.
    - [X] 对游戏界面进行判断(实验结论为使用自定义网络的Tensorflow的图像分类模型)

2.
    - [X] 从图像中获取各类按钮的坐标(实验结论为使用基于Tensorflow的ObjectDetectionAPI库)

3.
    - [X] 识别文字(实验结论为使用同样基于Tensorflow的PaddleOCR)

4.
    - [X] 利用adb获取串流(经过实验后结论为使用 scrcpy)

5.
    - [X] 利用adb实现点击(也使用scrcpy)

6.
    - [X] 用于清理智的操作逻辑程序

## 其他重要功能

- [X] 自动添加理智
- [X] 判断是否使用源石添加理智
- [X] 读取进度
- [X] 反馈模型检测精准度
- [ ] 降低CPU使用率
- [X] 支持GPU
- [X] 设置GPU显存限制
- [X] GPU动态显存
- [X] 多设备选择

---

## 待实现

- [ ] 提供adb设备操作界面
- [X] 显示系统信息(主要为GPU设备)
- [ ] 基于ffmpeg-python 对 H264解码进行GPU加速支持(cuvid,cuda,h264_amf,vaapi), 移除PyAV库

---

## 大饼

- [ ] 实现基建自动换班
- [ ] 实现指定不同关卡