import torch
from sklearn.cluster import KMeans
#这里需要两个部分，蒸馏池化+LSC，蒸馏池化的部分需要用到旧模型的信息
#而LSC的部分需要用到n个类别的K个代理的信息
#这里需要定义余弦相似性函数，以及NCA的计算方式



#由于要计算余弦相似度，这里要确保proxy和h同维
#或者proxy可以广播，以适应这种情况
#即确保proxy和h的形状为(nrows, 1)
def cosineSim(proxy, h): 
    return torch.matmul(proxy.T , h) / (torch.norm(proxy, p=2, dim=0, keepdim=True)*torch.norm(h, p=2, dim=0, keepdim=True))
                                                                                 


def NCAloss(eta, delta, true_class, y_hats, ):
    # 记住，这里是要对一个批次的数据进行NCA计算，所以会有最后的平均操作

    pass

def herdingStrategy():
    # 仿照icarl的兽群策略，用于样本集管理之中
    pass


class sampleSetManager():
    def __init__(self, DataSet = None):
        # 初始化的时候肯定没有数据
        self.dataset = DataSet
        
        pass
        # 样本集管理
    def appendSamples(self, TrainSet):
        # 利用兽群策略管理样本集
        pass
    
    pass


def proxy_initialize(k, trainSet):
    # trainset应当是[ [class , [tensors] ]
    '''
    Assert: `trainSet` 必须是由dataprocess的相关接口函数处理的数据, 才能保证接口兼容性
    '''
    # 应该在开始每个新任务的训练前就调用
    # 使用K-mens算法初始化K个proxys，之后由于损失中引入了proxy，后续的训练可以更新proxy
    # 要利用k-means算法，就需要知道整个新的训练样本集，这中间可能涉及数据的转换

    # 判断数据的所在地(CPU or GPU)
    # 可以直接使用sklearn的机器学习库进行KMeans
    # 直接接收tensor作为原始数据，返回array，需要重新转化回tensors
    
    seed = 42
    proxy_maker = KMeans(n_clusters=k,random_state=seed)
    proxy_dict = dict()
    on_gpu = False
    for class_, tensors in trainSet:
        if tensors.is_cuda:
            tensors = tensors.cpu()
            on_gpu = True
        res = proxy_maker.fit(tensors)
        proxys = res.cluster_centers_
        if on_gpu:
            proxys = proxys.cuda()
        proxy_dict[class_] = torch.tensor(proxys)

    return proxy_dict

def metric(model, testset):
    '''
    Input: 模型model与测试集testset
    Output: 当前在已见类别上的准确率
    '''
