import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from sklearn.cluster import KMeans

# 为了能顺利导入dataprocess模块
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, ".."))
from dataprocess import process
from model.PODNet import integratedMdl

# 绘制进度条
from tqdm import tqdm

# 这里要完成训练任务，首先要完成一个基本的接口函数：train
# 该接口函数接收参数，并调用basic_loop函数

# 鉴于之前的经验，这里决定在使用训练取批次数据的时候再把数据转到gpu


# 需要规定：
# on_gpu, batch_size, inc_step, data_name, train_with_pretrained_inc, load_pretrained_path, save_pretrained_path 
# SGD_learning_rate, decay_rate, SGD_momentum, train_ratio, SGD_momentum
def train(**kargs):
    #  需要参考原做法，结合cosine annealing scheduling
    # 首先进行参数解析
    # 这些交给主函数调用的时候处理即可
    data_name = kargs["data_name"] if "data_name" in kargs else "cifar100"
    max_epoch_num = kargs["epoch_num"] if "epoch_num" in kargs else 100
    train_with_pretrained_inc = kargs["train_with_pretrained_inc"] if "train_with_pretrained_inc" in kargs else False
    load_pretrained_path = kargs["load_pretrained_path"] if "load_pretrained_path" in kargs else None



    base_dataset = process.generalDataSet(data_name, "train") if "train_ratio" not in kargs else \
                   process.generalDataSet(data_name, "train", train_ratio = kargs["train_ratio"]) 
    # 首先根据信息定义模型，并尽可能在初始化时就将模型转移到gpu上
    model = integratedMdl(train_with_pretrained_inc,load_pretrained_path)
    # 定义优化器
    optimizer = torch.optim.SGD(lr = kargs["SGD_learning_rate"], weight_decay=kargs["decay_rate"])
    learned_classes = 0 #已经学过的类别数目
    # 进入首次训练前，先进行proxy的初始化，调用pretrain
    pre_train(model,[i for i in range(50)],base_dataset)
    # 首先进入首次训练
    basic_task_set = base_dataset.getIncData4Train(base=True)

    
    



    # 注意余弦退火策略与早停策略

    
    # 开始准备优化器，做训练前的准备
    for epoch in range(max_epoch_num):
        pass
    try:
        pass
    except:
        pass


    pass
    

# 在真正的训练开始前运行，初始化proxys
# python默认按引用传值
def pre_train(model: integratedMdl, newClassLst: list, baseData: process.generalDataSet):
    '''
    model: 还没初始化新类别proxy的模型(integratedMdl类型)
    newClassLst: 新增类别列表，例如[0,1,2], 即新增类别0,1,2(按顺序,指的并不是实际类别标签)
    baseData: 存储整个训练集的基础数据集(generalDataSet类型)
    '''
    # - 这里只需要执行推理，但最好，在这里就把数据都转移到gpu上
    #   这样，最后得到的proxy张量也都会在gpu上
    # - 这里的model已经转移到gpu了(若可用)
    new_proxys = {}
    batch_size = 64
    clusterer = KMeans(n_clusters=model.curModel.proxyNums,random_state=42)

    # 对每个类别进行推理，得到每个类别的feature的k个聚类中心，作为初始化的proxy
    for newClass in newClassLst:
        subset = baseData.getSpecificData(newClass)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        # 开始进行推理，并获取其输出
        feas = []
        with torch.no_grad():
            for images , _ in loader:
                images = images.to("cuda" if torch.cuda.is_available() else "cpu")
                output = model.curModel(images)
                # output形状(batchsize, feas, 1)
                # Kmeans接口, 每一行代表一个样本，每一列代表一个特征，所以应该把它转成(batchsize, feas)
                output = torch.squeeze(output).to("cpu")
                feas.append(output)

            forCluser = torch.cat(feas, dim=0)
            clusterer.fit(forCluser)
            # 这里的proxy形状还需要再商榷
            proxys = torch.tensor(clusterer.cluster_centers_).to("cuda" if torch.cuda.is_available() else "cpu")
            # 我这里的proxy，形状一定要是(10, feas, 1)，最后一维一定是1！
            proxys.unsqueeze(-1)
            new_proxys[newClass] = proxys
            
    model.curModel.append_proxys(new_proxys)




# # 规定模型的
# def basic_loop():

#     pass






if __name__ == '__main__':

    # 定义数据转换
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose(
        [transforms.ToTensor()])
    # 通过compose操作，将dataset变成(tensor-->image, int-->class)

    # 导入训练集
    trainset = torchvision.datasets.CIFAR100(root='./dataset', train=True,
                                            download=True, transform=transform)
    Strainset = Subset(trainset, list(range(1000)))
    trainloader = torch.utils.data.DataLoader(Strainset, batch_size=32,
                                            shuffle=True)
    # 这里有个问题，为什么用不了多线程

    # 类别标签
    # classes = trainset.classes

    # 打印数据集大小
    print('训练集大小:', len(trainset))
    import cv2
    # cv2.imshow("河狸",trainset[0][0].numpy().transpose(1,2,0))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 可以看出，图片确实很小，所以肯定得做图片上的变换
    for batch_idx,(datas, labels) in enumerate(trainloader):
        # x是tensor， labels是字符串类型
        print(datas)
        
        print(labels)