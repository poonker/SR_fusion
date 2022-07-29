# SR_mfusion

### 本项目收集SR（super resolution）研究领域中多个成熟的预训练模型，统一调用运行，方便评估对比(A combine operation for SR models)。

### 环境配置(Requirements)
```bash
pip install -r requirements.txt
```

### 数据准备(data prepare)
请将任意图片放入test_image文件夹内,目录结构参考下方(put any picture into test_image folder)。

### 目录结构(directory structure)
- 计划所有模型获取test_image文件夹内图片，运行结果汇总至test_result文件夹内。
- 注意：以下有any前缀文件皆未指定名称及内容。
```
Bicubic/
SRCNN/
FSRCNN/
VDSR/
DRCN/
RDN/
SAN/
SRGAN/
    └── config.py
    └── srgan.py
    └── train.py
    └── vgg.py
    └── model
          └── vgg19.npy
USRGAN/
any_model/
test_image/
    └── any_image.png
test_result/
    └── any_model
          └── any_result.txt
          └── any_image_result.png   
main.py
test.py
config.py
requirements.txt
README.md
```
### 运行(run)
```bash
python main.py
```

### 计划中的模型链接地址(reference list)
* [SRCNN](https://github.com/yjn870/SRCNN-pytorch)
* [FSRCNN](https://github.com/yjn870/FSRCNN-pytorch)
* [VDSR](https://github.com/twtygqyy/pytorch-vdsr)
* [DRCN]()
* [RDN](https://github.com/hengchuan/RDN-TensorFlow)
* [SAN](https://github.com/daitao/SAN)
* [SRGAN](https://github.com/leftthomas/SRGAN)
* [USRGAN](https://github.com/cszn/USRNet)


### 结果展示(result)
None
