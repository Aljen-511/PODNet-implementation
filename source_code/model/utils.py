import torch
from sklearn.cluster import KMeans
from torch.utils.data import ConcatDataset

#这里需要两个部分，蒸馏池化+LSC，蒸馏池化的部分需要用到旧模型的信息
#而LSC的部分需要用到n个类别的K个代理的信息
#这里需要定义余弦相似性函数，以及NCA的计算方式



#由于要计算余弦相似度，这里要确保proxy和h同维
#或者proxy可以广播，以适应这种情况
#即确保proxy和h的形状为(nrows, 1)
#这里事实上确保proxy是(batch,feas,1), h是(feas,1)即可
def cosineSim(proxy, h): 
    new_h = h.view(-1,1)
    return torch.matmul(new_h.T , proxy) / (torch.norm(proxy, p=2, dim=-2, keepdim=True)*torch.norm(new_h, p=2, dim=-2, keepdim=True))
                                                                                 


def NCAloss(eta, delta, true_classes, y_hats, class_lst):
    '''
    eta: 参照论文里NCA公式里的eta
    delta: 参照论文里NCA公式里的delta, 也即margin
    true_classes: 一个批次中每个样本的真实预测类别(在训练的时候给出), (batch, 1)
    y_hats: 一个批次中, 每个样本对每个类别的评分(可理解为概率, 由LSC分类器给出) list[(b)]
    '''
    # 记住，这里是要对一个批次的数据进行NCA计算，所以会有最后的平均操作

    y_hats = torch.stack(y_hats,axis = -1).squeeze()
    # 这里通过布林操作筛选出y_hat_y, class_lst是一个列表list[c], 先转成tensor, 
    # 再堆叠batch次, 注意, 需要转移到gpu
    all_class = torch.stack([torch.tensor(class_lst)]*true_classes.shape[0],dim=0).squeeze()
    all_class = all_class.to("cuda" if torch.cuda.is_available() else "cpu")
    # 这里all_class形状为(batch, class)
    true_idx = all_class == true_classes
    y_hat_y = y_hats[true_idx].view(-1,1)
    # 将真实类别的值剔除掉, 方便后续的求和操作
    y_hats[true_idx] = 0.0
    y_hats = eta*torch.exp(y_hats)

    L_LSC = -(eta*(y_hat_y-delta) - torch.log(torch.sum(y_hats, dim = 1)).view(-1,1))
    L_LSC = torch.clamp(L_LSC, min=0.)
    return torch.mean(L_LSC)




def herdingStrategy(samples:torch.Tensor, M=20):
    '''参考iCarl的论文'''
    # 仿照icarl的兽群策略，用于样本集管理之中
    # 这里接收的输入是批量的张量, 是经过网络输出的特征, 形状是(sample_nums, feas)
    # TODO: 利用herding策略返回选出的样本下标 √
    with torch.no_grad():
        chosen_idx = []
        mu = torch.mean(samples,dim=0)
        phi_sofar = torch.full(mu.shape, 0.,dtype=mu.dtype)
        for k in range(M):
            dis_metric = mu - 1./(k+1) * (samples + phi_sofar)
            dis_metric = torch.norm(dis_metric,dim=1)
            min_idx = torch.argmin(dis_metric).item()
            chosen_idx.append(min_idx)
            phi_sofar += samples[min_idx]
    
    return chosen_idx



class sampleSetManager():
    def __init__(self, DataSet = None):
        # 初始化的时候肯定没有数据
        self.dataset = DataSet

        # 样本集管理
    def appendSamples(self, TrainSet):
        # 利用兽群策略管理样本集
        '''TrainSet必须是dataset类型, 这样才方便后续的操作'''
        if self.dataset is None:
            self.dataset = TrainSet
        else:
            self.dataset = ConcatDataset([self.dataset, TrainSet])
    


# 该函数作废, 因为train.loop中的pre_train函数起到了相同的作用
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


