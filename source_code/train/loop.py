import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader, ConcatDataset
from sklearn.cluster import KMeans

# 为了能顺利导入dataprocess模块
import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, ".."))
from dataprocess import process
from model.PODNet import integratedMdl
from test.test import acc_loop
from model.utils import herdingStrategy
import numpy as np
import copy


# 绘制进度条
from tqdm import tqdm

# 这里要完成训练任务，首先要完成一个基本的接口函数：train
# 该接口函数接收参数，并调用诸多loop函数

# 鉴于之前的经验，这里决定在训练取批次数据的时候再把数据转到gpu


# 需要规定：
# on_gpu, batch_size, inc_step, data_name, train_with_pretrained_inc, load_pretrained_path, save_pretrained_path 
# SGD_learning_rate, decay_rate, SGD_momentum, train_ratio, SGD_momentum, T_max(余弦退火周期), max_epoch
def train_and_evaluate(**kargs):
    #  需要参考原做法，结合cosine annealing scheduling
    # 首先进行参数解析
    # 这些交给主函数调用的时候处理即可
    ######################################
    torch.autograd.set_detect_anomaly(True)
    ######################################
    data_name = kargs["data_name"] if "data_name" in kargs else "cifar100"

    

    print(f"Loading data from {data_name}......")
    base_trainset = process.generalDataSet(data_name, "train") if "train_ratio" not in kargs else \
                   process.generalDataSet(data_name, "train", train_ratio = kargs["train_ratio"])
    ##########################
    kargs["train_ratio"] = 0.9
    ##########################
    base_valset = process.generalDataSet(data_name, "val") if "train_ratio" not in kargs else \
                  process.generalDataSet(data_name, "val",train_ratio = kargs["train_ratio"])

    base_testset = process.generalDataSet(data_name, "test")

    print("Data Loading done! Initializing the neccessary configurations for training......")

    
    # 首先根据信息定义模型，并尽可能在初始化时就将模型转移到gpu上
    model = integratedMdl()
    accuracy_metric = []
    # 开始基础任务的训练
    learned_classes = 0 #已经学过的类别数目
    # 进入首次训练前，先进行proxy的初始化，调用pretrain
    
    print("Start the basic training(50 classes)...")
    pre_train(model,[i for i in range(50)],base_trainset)
    basic_loop(model=model,
               lr_=kargs["SGD_learning_rate"],
               weight_dec=kargs["decay_rate"],
               momentum_=kargs["SGD_momentum"],
               baseTrainset=base_trainset,
               baseValset=base_valset,
               T_max_ = kargs["T_max"],
               max_epoch=kargs["max_epoch"],
               batch_size_=kargs["batch_size"])
    after_train(model, base_trainset,[i for i in range(50)] )
    accuracy_metric.append(acc_loop(model,base_testset,50))
    learned_classes += 50
    print(f"Basic traing is done. Average accuracy on testset is : {accuracy_metric[-1]:.6f}")
    
    
    inc_step = kargs["inc_step"]
    num_tasks = int(np.ceil(50/inc_step))
    print(f"Start the increase training.\nincrease step: {inc_step}\ttasks number: {num_tasks}")
    for task in range(num_tasks):
        # TODO: 将之前保留的旧样本并入训练集
        # 开始每个增量阶段的训练
        print(f"task {task}")
        pre_train(model, [i for i in range(learned_classes, learned_classes+inc_step)], base_trainset)
        basic_loop(lr_=kargs["SGD_learning_rate"],
                   weight_dec=kargs["decay_rate"],
                   momentum_=kargs["SGD_momentum"],
                   baseTrainset=base_trainset,
                   baseValset=base_valset,
                   T_max_=kargs["T_max"],
                   max_epoch=kargs["max_epoch"],
                   batch_size_=kargs["batch_size"],
                   inc_classesLst=[i for i in range(learned_classes, learned_classes+inc_step)])
        after_train(model=model, 
                    baseTrainset=base_trainset, 
                    new_classLst=[i for i in range(learned_classes, learned_classes+inc_step)])
        accuracy_metric.append(acc_loop(model=model,baseTestset=base_testset, inc_step=inc_step))
        learned_classes += inc_step
        cur_avg_acc = np.mean(np.array(accuracy_metric))
        print(f"Current average class accuracy: {cur_avg_acc:.6f}")
    
    #完成增量学习，开始打印评估信息
    

    
    ##### TODO: 1. 实现两种循环(一种给初次训练任务，一种给增量训练任务) /6.1 √
    #           2. 实现指标(平均类别准确率) /6.1 √
    #           3. 实现test的功能，因为现在每一次增量任务之后都要计算准确率，所以需要写test的循环 /6.1 √
    #           4. 进度条 /6.1 √

    #metric先不要独立出来，因为没有必要
    


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
    batch_size = 128
    clusterer = KMeans(n_clusters=model.curModel.proxyNums,random_state=42)

    # 对每个类别进行推理，得到每个类别的feature的k个聚类中心，作为初始化的proxy
    for newClass in newClassLst:
        subset = baseData.getSpecificData(newClass)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
        # 开始进行推理，并获取其输出
        feas = []
        with torch.no_grad():
            for images , class_ in loader:
                images = images.to("cuda" if torch.cuda.is_available() else "cpu")
                output = model.curModel(images)

                # output形状(batchsize, feas, 1)
                # Kmeans接口, 每一行代表一个样本，每一列代表一个特征，所以应该把它转成(batchsize, feas)
                output = torch.squeeze(output).to("cpu")
                feas.append(output)

            forCluser = torch.cat(feas, dim=0)
            clusterer.fit(forCluser)
            # 这里的proxy形状还需要再商榷
            # 这里一定要确保数据类型为float, 否则后续无法和神经网络的输出做运算
            proxys = torch.tensor(clusterer.cluster_centers_,dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
            # 我这里的proxy，形状一定要是(10, feas, 1)，最后一维一定是1！
            proxys.unsqueeze(-1)
            # 这里debug时排查出问题所在, class_是一个tensor, 不能拿tensor作为字典的键, 否则后续无法索引
            # new_proxys[class_] = proxys #旧写法
            # 确保索引为整数
            new_proxys[int(class_[0].item())] = proxys
            
    model.curModel.append_proxys(new_proxys)

# 在训练完成之后，在模型中保留一定的旧类别样本
def after_train(model: integratedMdl, 
                baseTrainset:process.generalDataSet,
                new_classLst
                ):
    '''
    model是已经经过训练之后的模型了
    new_classLst是本次增量学习的类别序号
    baseTrainset具有所有类别的训练数据
    '''
    batch_size_ = 128
    num_workers_ = 2
    # TODO: 调用样本集管理策略，保留某些旧样本 √
    # 所以这里需要在已经训练完的模型上进行推理，得到特征，之后再调用
    # model里的samplemanager，运用herding策略保留样本
    for new_class in new_classLst:
        subset = baseTrainset.getSpecificData(new_class)
        dataloader = DataLoader(subset, batch_size=batch_size_, num_workers=num_workers_, shuffle=False)
        with torch.no_grad():
            features = []
            for image, _ in dataloader:
                image = image.to("cuda" if torch.cuda.is_available() else "cpu")
                output = model.curModel(image)
                # 这里转移回cpu是为了后续操作的统一
                output = torch.squeeze(output).to("cpu")
                features.append(output)
            samples = torch.cat(features,dim=0).squeeze()
            chosen_idx = herdingStrategy(samples)
            model.setManager.appendSamples(Subset(subset, chosen_idx))  

    pass


# 规定模型的基本循环
def basic_loop(model:integratedMdl,
               lr_, weight_dec, momentum_, 
               baseTrainset:process.generalDataSet, 
               baseValset:process.generalDataSet, 
               T_max_, max_epoch, batch_size_, 
               inc_classesLst = None
               ):
    '''
    model: 待训练的模型
    lr_: SGD优化器的学习率
    weight_dec: SGD优化器的衰减权重
    baseTrainset: 基本训练集
    baseValset: 基本验证集
    T_max: 余弦退火策略的周期
    max_epoch: 最多的训练代数
    batch_size: 训练的批数
    inc_classesLst: 当前增量学习的类别序号列表, 长度即为inc_step,
                    默认为None时, 说明这是一个base训练循环
    '''
    # TODO: 
    #   1. 早停策略 √
    #   2. 余弦退火策略 √

     
    # 创建数据加载器
    num_workers_ = 4
    
    # 这里不要忘了, 要将之前保留的旧类别样本合并进来
    if inc_classesLst is None:
        basic_task_set = baseTrainset.getIncData4Train(base = True)
    else:
        basic_task_set = ConcatDataset([baseTrainset.getIncData4Train(inc_nums=len(inc_classesLst)),
                                        model.setManager.dataset])
    
    
    basic_val_set = baseValset.getIncData4Train(base=True) if inc_classesLst is None else \
                    baseValset.getIncData4Train(inc_nums=len(inc_classesLst))
    
    train_dataloader = DataLoader(basic_task_set,shuffle=True, batch_size=batch_size_,num_workers=num_workers_)
    val_dataloader = DataLoader(basic_val_set, shuffle=False, batch_size=batch_size_, num_workers=num_workers_)
    # 创建优化器，并绑定余弦退火学习策略
    optimizer = torch.optim.SGD(lr=lr_, weight_decay=float(weight_dec), momentum=momentum_, params=model.curModel.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=T_max_)
    # !这里可能存在一些问题...
    best_acc = -1.0
    for epoch in range(max_epoch):
    # for epoch in tqdm(range(max_epoch),desc=f"Epoch nums",unit="epoch",leave=False):
        # for image, label in train_dataloader:
        for image, label in tqdm(train_dataloader, desc=f"training Epoch {epoch+1}/{max_epoch}",unit="itr",leave=False):
            # 这里尽可能使其转移到gpu上
            image = image.to("cuda" if torch.cuda.is_available() else "cpu")
            label = label.to("cuda" if torch.cuda.is_available() else "cpu")
            optimizer.zero_grad()
            y_output = model.curModel(image)
            # 若是增量学习任务，则还需要在旧模型上推理一遍，以计算损失函数

            # TODO: 这里的损失计算函数还有一点问题y
            if model.oldModel is not None:
                with torch.no_grad():
                    model.oldModel(image)
            
            y_c_lst = model.curModel.LSC_Classifier(y_output, predict_only=False)
            loss = model.PODloss(true_classes=label.view(-1,1), y_hat_c=y_c_lst)
            loss.backward()
            # 这里在每一个iteration都更新一次学习率
            optimizer.step()
            scheduler.step()

        # 完成一个epoch的训练之后，应该在val集上检验性能，以便执行早停
        # TODO: 早停 6.1 √
        true_predict = 0
        totol_predict = 0
        with torch.no_grad():
            # for image, label in val_dataloader:
            for image, label in tqdm(val_dataloader, desc=f"evaluating on Epoch {epoch+1}/{max_epoch}",unit="itr",leave=False):
                image = image.to("cuda" if torch.cuda.is_available() else "cpu")
                y_output = model.curModel(image)
                _, result = model.curModel.LSC_Classifier(y_output)
                # 注意，这里的result应该是批量的张量，要对其形状进行具体的分析-->并不是,而是list[tensors]
                # 之后再进行比较，得到预测正确的数量
                # TODO: 比较预测结果与真实标签 √
                # result的形状应该是(batchsize, 1), 而且是tensor --> 
                result = result.to("cpu").numpy()
                label = label.view(-1,1).numpy()
                true_predict += np.sum(result == label)
                totol_predict += result.shape[0]

        # 补完得差不多了，继续努力🥳
        # 粗糙的早停
        if true_predict/totol_predict < best_acc and epoch > 30:
            print(f"Stop early at Epoch{epoch+1}! Final accuracy: {true_predict/totol_predict:.6f}")
            break
        else:
            best_acc = true_predict/totol_predict if true_predict/totol_predict > best_acc else best_acc
            print(f"acc at epoch {epoch+1}: {best_acc:.6f}")

    # 做深拷贝, 将当前已训练完的模型进行复制
    model.oldModel = copy.deepcopy(model.curModel)

    pass




'''
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

'''

