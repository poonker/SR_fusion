import argparse
import glob
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import convert_rgb_to_y
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


def train(args):
    # 'w' 创建文件，已经存在的文件会被覆盖掉
    h5_file = h5py.File(args.output_path, 'w')

    lr_patches = []
    hr_patches = []

# glob模块用来查找文件目录和文件，并将搜索的到的结果返回到一个列表中,*代表0个或多个字符
# .format将括号内信息填入{}
    for image_path in sorted(glob.glob('{}/*'.format(args.images_dir))):
        # 将照片转换为RGB通道
        # PIL.Image中open是加载图像，convert用于转换图像模式
        """
        convert参数如下：
        1 ------------------（1位像素，黑白，每字节一个像素存储）
        L ------------------（8位像素，黑白）
        P ------------------（8位像素，使用调色板映射到任何其他模式）
        RGB------------------（3x8位像素，真彩色）
        RGBA------------------（4x8位像素，带透明度掩模的真彩色）
        CMYK--------------------（4x8位像素，分色）
        YCbCr--------------------（3x8位像素，彩色视频格式）
        I-----------------------（32位有符号整数像素）
        F------------------------（32位浮点像素）
        """
        hr = pil_image.open(image_path).convert('RGB')
        # 取放大倍数的倍数
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        # 图像大小调整，预处理
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        # 低分辨率图像缩小
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        # 低分辨率图像放大
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        # 转换为浮点并取ycrcb中的y通道
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        for i in range(0, lr.shape[0] - args.patch_size + 1, args.stride):
            for j in range(0, lr.shape[1] - args.patch_size + 1, args.stride):
                lr_patches.append(lr[i:i + args.patch_size, j:j + args.patch_size])
                hr_patches.append(hr[i:i + args.patch_size, j:j + args.patch_size])

    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    # 创建数据集
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)

    h5_file.close()


def eval(args):
    h5_file = h5py.File(args.output_path, 'w')

    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')

    for i, image_path in enumerate(sorted(glob.glob('{}/*'.format(args.images_dir)))):
        hr = pil_image.open(image_path).convert('RGB')
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        lr = lr.resize((lr.width * args.scale, lr.height * args.scale), resample=pil_image.BICUBIC)
        hr = np.array(hr).astype(np.float32)
        lr = np.array(lr).astype(np.float32)
        hr = convert_rgb_to_y(hr)
        lr = convert_rgb_to_y(lr)

        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)

    h5_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--patch-size', type=int, default=33)
    parser.add_argument('--stride', type=int, default=14)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    if not args.eval:
        train(args)
    else:
        eval(args)
