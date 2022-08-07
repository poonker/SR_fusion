import h5py
import numpy as np
from torch.utils.data import Dataset
"""
训练模型一般都是先处理 数据的输入问题 和 预处理问题。Pytorch提供了几个有用的工具：torch.utils.data.Dataset 类和 torch.utils.data.DataLoader 类
流程是先把原始数据转变成 torch.utils.data.Dataset 类，随后再把得到的 torch.utils.data.Dataset 类当作一个参数传递给 torch.utils.data.DataLoader 类，
得到一个数据加载器，这个数据加载器每次可以返回一个 Batch 的数据供模型训练使用。
在 pytorch 中，提供了一种十分方便的数据读取机制，即使用 torch.utils.data.Dataset 与 Dataloader 组合得到数据迭代器。
在每次训练时，利用这个迭代器输出每一个 batch 数据，并能在输出时对数据进行相应的预处理或数据增广操作。
如果我们要自定义自己读取数据的方法，就需要继承类 torch.utils.data.Dataset ，并将其封装到DataLoader 中。
"""
# args.train_file作为Dataset参数被传入，下面连个类只在train中用到
"""
self指的是实例Instance本身,在Python类中规定，函数的第一个参数是实例对象本身，并且约定俗成，把其名字写为self
也就是说，类中的方法的第一个参数一定要是self，而且不能省略。
Python中的super(Net, self).__init__()是指首先找到Net的父类（比如是类NNet），然后把类Net的对象self转换为类NNet的对象，
然后“被转换”的类NNet对象调用自己的init函数，其实简单理解就是子类把父类的__init__()放到自己的__init__()当中，
这样子类就有了父类的__init__()的那些东西。
"""
class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

# with as 语句操作上下文管理器（context manager），它能够帮助我们自动分配并且释放资源。
# np.expand_dims:用于扩展数组的维度
# 'r'是模式表示只读，文件必须存在
    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            # 维度变换目的是为了满足模型输入
            return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
