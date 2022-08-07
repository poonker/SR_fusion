from torch import nn
"""
nn.Conv2d()类作为二维卷积的实现，第一个参数是输入channels值
第二个参数是期望的输出channels数
第三个参数kernel_size是卷积核大小
Padding即所谓的图像填充，后面的int型常数代表填充的多少（行数、列数），默认为0。
需要注意的是这里的填充包括图像的上下左右，以padding = 1为例，若原始图像大小为32x32，那么padding后的图像大小就变成了34x34，而不是33x33。
"//"取整除 - 返回商的整数部分（向下取整）
nn.ReLU中inplace为True，将会改变输入的数据 ，否则不会改变原输入，只会产生新的输出
"""
# 三个卷积核
class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        # 第一层卷积：卷积核尺寸9×9(f1×f1)，卷积核数目64(n1)，输出64张特征图
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        # 第二层卷积：卷积核尺寸1×1(f2×f2)，卷积核数目32(n2)，输出32张特征图
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        # 第三层卷积：卷积核尺寸5×5(f3×f3)，卷积核数目1(n3)，输出1张特征图即为最终重建高分辨率图像
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
