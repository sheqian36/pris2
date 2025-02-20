# pris2
AI课程任务2 - 多模态融合之文本指导的图像上色，已实现推理


## 使用说明

1. 创建并激活虚拟环境：

```python
conda create -n pris2 python=3.8
```

```
pip install git+https://github.com/sheqian36/pris2.git
```
or
```
git clone https://github.com/sheqian36/pris2.git
cd pris2
pip install ./pris2
```
## 权重加载

把预训练权重 largedecoder.pth 放到 ./pris2 目录

权重下载：[谷歌网盘](https://drive.google.com/file/d/1xA-4hzY-zBxtxywVw_9y2u17ExenRVju/view)

## 启动说明
1. 导入并启动测试：

```
import pris2
img_path = "L-CoDer.jpg"

caption = "There is a blue car in the middle of the road."

output_path = "colorized.jpg"

pris2.color(img_path, caption, output_path)

```
或者
```
python demo.py
```
可以更换需要上色的图像和测试不同的文本

## 依赖项
- Python 3.8
- Conda

## 引用
该项目基于 ``https://github.com/changzheng123/L-CoDer``进行封装，可以给原仓库一个 :star: