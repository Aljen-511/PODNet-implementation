
# 因为读取的数据可能较大，所以，不能一口气将数据都读入内存，这一点不用担心，torch库已经考虑到这一点了

# 因为imagenet100的数据集无法通过torchvision加载，
# 所以可能需要手动放置在数据集里(手动放在服务器上了)


# 需要处理为统一的dataset

# 训练与推理时才需要进一步转换为dataloader
import os
import torchvision
from torchvision import transforms
import torch
from torch.utils.data import Subset


# 需要区分训练集，测试集，验证集
# 需要对数据集进行划分(按类别)，以便于后续的使用
# 参考论文的测试，论文首先使用前一半的类别进行训练，时候再在后一半的类别上进行增量式学习
class generalDataSet:
    def __init__(self, dataName, type = "train", **kargs):
        assert(type in ["train", "val", "test"]), "Unknown Arguments: 使用了未知的任务参数"+type
        assert(dataName in ["cifar100", "imagenet100"]), "你引用了一个不可用的数据集名称："+dataName+"，可用数据集有cifar100,imagenet100"
        self.dataset = None
        self.type = type
        self.dataName = dataName
        self.taken_class_nums = 0
        if dataName == "cifar100":
            # 这里已经是一个dataset了，每一个item的结构为(iamge:tensor, class:int)
            # 对cifar100，需要特殊处理， 因为其只能加载训练集和测试集，所以需要在训练集上进行随机采样，得到验证集
            # 这里假设原本的cifar100已经过随机排序了，所以直接截取后面的部分作为验证集
            self.dataDir = os.path.join(os.getcwd(),"dataset/")
            if type == "train":
                # 对数据集做一定的变换，并进行加载
                transform = transforms.Compose( [transforms.ToTensor(),
                                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ColorJitter(brightness=63 / 255)
                                                 ]) # 需要参照其实现进行修改
                trainset = torchvision.datasets.CIFAR100(self.dataDir, train=True, download=True, transform=transform)
                if "train_ratio" in kargs:
                    division_pt = int(kargs["train_ratio"]*len(trainset))
                else:
                    division_pt = int(0.8*len(trainset))
                self.dataset = Subset(trainset, range(0, division_pt))
                
            elif type == "val":
                transform = transforms.Compose( [transforms.ToTensor(),
                                                transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                                                transforms.RandomCrop(32, padding=4),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ColorJitter(brightness=63 / 255)
                                                 ])
                valset = torchvision.datasets.CIFAR100(self.dataDir, train=True, download=True, transform=transform)
                if "train_ratio" in kargs:
                    division_pt = int(kargs["train_ratio"]*len(valset))
                else:
                    division_pt = int(0.8*len(valset))
                self.dataset = Subset(valset, range(division_pt, len(valset)))               
            
            elif type == "test":
                # test无需做过多的变换
                transform = transforms.Compose( [transforms.ToTensor(),
                                                 transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
                                                 ])
                self.dataset = torchvision.datasets.CIFAR100(self.dataDir, train=False, download=True, transform=transform)


        elif dataName == "imagenet100":
            self.dataDir = "/autodl_hub/tmpdata/Imagenet100"
            if type == "train" or type == "val":
                transform = transforms.Compose( [transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                transforms.RandomResizedCrop(224),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ColorJitter(brightness=63 / 255)
                                                 ]) # 需要参照其实现进行修改
                self.dataset = torchvision.datasets.ImageNet(root=self.dataDir, split=type, transform=transform, download=False)
            
            elif type == "test":
                transform = transforms.Compose( [transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                transforms.Resize(256),
                                                transforms.CenterCrop(224)
                                                ]) # 需要参照其实现进行修改
                self.dataset = torchvision.datasets.ImageNet(root=self.dataDir, split=type, transform=transform, download=False)
        
        self.sortedByClass()

    # 使得处理后的数据集，同一类别的数据都按顺序聚在一起
    def sortedByClass(self):
        if self.dataName == "cifar100":
            self.class_order = [  # 三手实现🤣，PODNET原作者从icarl参考的cifar100类别顺序实现
                        87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18,
                        24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59,
                        25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
                        60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7,
                        34, 55, 54, 26, 35, 39
                        ] # 有100类，改变训练集中类别出现的顺序
        else:
            self.class_order = [i for i in range(100)] # imagenet就不需要特殊照顾了   
        # 这里创建索引，之后可以直接根据索引返回对应的数据集
        self.class_idx = {}
        for item in self.class_order:
            self.class_idx[item] = []
        for idx in range(len(self.dataset)):
            self.class_idx[self.dataset[idx][1]].append(idx)
            


    # 这里默认数据集里有100个类别
    def getIncData4Train(self, inc_nums = None, base=False):
        
        assert (inc_nums is None and not base) or (inc_nums is not None and base), "错误使用getIncData"
        if inc_nums is not None and self.taken_class_nums != 50:
            raise Exception("逻辑错误: 在增量式地攫取数据集时, 首次攫取的类数不为50")
        merged_idx = []
        # 返回前50个类别，供初次学习
        if base:
            for i in range(50):
                merged_idx.extend(self.class_idx[self.class_order[i]])
            # 标识已经拿走的类别数量
            self.taken_class_nums = 50
        # 按类别步长(inc_nums)，返回每次增量任务的数据集
        else:
            for i in range(self.taken_class_nums, self.taken_class_nums+inc_nums):
                merged_idx.extend(self.class_idx[self.class_order[i]])
            self.taken_class_nums += inc_nums

        # 按索引返回供增量学习的数据子集，注意，是dataset格式
        return Subset(self.dataset, merged_idx)

    # 这里获取某个单独类别的子集
    def getSpecificData(self, class_order):
        assert class_order >= 0 and class_order < 100, "类别索引超出范围，数据集只有100个类别"

        specific_idx = self.class_idx[self.class_order[class_order]]
        return Subset(self.dataset, specific_idx)


# TODO: 根据参考实现，修改各个transform -24.5.30 √
# TODO: 将数据按类别顺序排列整齐 -24.5.31 √




