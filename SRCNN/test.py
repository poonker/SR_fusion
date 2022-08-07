import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr

"""
一个python文件通常有两种使用方法，第一是作为脚本直接执行，第二是 import 到其他的 python 脚本中被调用（模块重用）执行。
因此 if __name__ == 'main': 的作用就是控制这两种情况执行代码的过程，
在 if __name__ == 'main': 下的代码只有在第一种情况下（即文件作为脚本直接执行）才会被执行，而 import 到其他脚本中是不会被执行的。
"""
if __name__ == '__main__':
    # argparse是一个Python模块,ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
    parser = argparse.ArgumentParser()
    # 增加参数
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    # ArgumentParser 通过 parse_args() 方法解析参数。
    # 至此，args拥有了上述信息参数。
    args = parser.parse_args()

    # 如果网络的输入数据维度或类型上变化不大，也就是每次训练的图像尺寸都是一样的时候，设置 torch.backends.cudnn.benchmark = true 可以增加运行效率；
    # 如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
    cudnn.benchmark = True

    # mytensor = my_tensor.to(device)
    # 这行代码的意思是将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
    # 这句话需要写的次数等于需要保存GPU上的tensor变量的个数；一般情况下这些tensor变量都是最开始读数据时的tensor变量，后面衍生的变量自然也都在GPU上
    # 这两句主要是将运算放在GPU上运行
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SRCNN().to(device)

    # model.parameters()与model.state_dict()是Pytorch中用于查看网络参数的方法。一般来说，前者多见于优化器的初始化，后者多见于模型的保存
    state_dict = model.state_dict()

    #  Load all tensors onto the CPU, using a function:
    #  torch.load('tensors.pt', map_location=lambda storage, loc: storage)
    # .items()返回可遍历的(键, 值) 元组数组。
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        # state_dict中存放了key和value的值，.keys()是获取key
        if n in state_dict.keys():
            # 调用copy_()的对象是目标tensor，参数是复制操作from的tensor，最后会返回目标tensor
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    # 在model(test_datasets)之前，需要加上model.eval(). 否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有batch normalization层所带来的的性质。
    # 在做one classification的时候，训练集和测试集的样本分布是不一样的，尤其需要注意这一点。
    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
    image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', '_srcnn_x{}.'.format(args.scale)))
