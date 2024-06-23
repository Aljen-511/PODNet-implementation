# 有个问题：数据放在哪？
# 保留的旧样本集应该放在哪？
# 放在PODNet部分吗？
# 目前想到的比较好的方法是：
# 1.再写一个model类，把PODNet放在里面，同时把旧样本集也放在里面
# 2.model类里保留一个旧模型，因为要进行训练
# 3.损失函数写在utils.py里
# 4.旧样本的管理策略也写在utils.py里
# 5.因为是K代理策略，所以代理的计算写在utils.py里
# 6.旧样本集合的大小怎么定？根据icarl的策略决定吗
# 8.模型的存储要考虑样本集
# --------24.5.28---xie
# 1. 直接把样本集放在模型类里
# 2. 增加一个抽象的model类，把模型和数据集都放在里面
# 3. 为了方便做消融实验，还是需要分接口

from model import ResNet
from model import utils
from torch import nn
import torch

global data2conv 
data2conv = {
    "cifar100":"resnet34",
    "imagenet100":"resnet18"
}

class integratedMdl():
    def __init__(self, load_checkpoint = False, load_path="",data_name = "cifar100"):
        # 这里如果决定要使用预训练模型，则要做好模型载入的工作
        # 在init的时候就完成张量转移
        # 同时，依然要注意判断：数据是在cpu上还是在gpu上
        if load_checkpoint:
            # TODO: 做好预训练模型的载入 X
            # 这里不需要有预训练模型, 最多就是需要有一个预训练的backbone
            # 但是, 由于框架已经定好了, 所以先不修改

            pass
        else:
            self.curModel = PODNet(pretrainedBackbone=True, backboneType=data2conv[data_name])
            if torch.cuda.is_available():
                self.curModel  = self.curModel.to("cuda")
            self.oldModel = None
            self.setManager = utils.sampleSetManager() #采用某种样本集管理策略
        
        self.eta = 1.0 # 可以学习的超参数，但是原论文的实现中，似乎没有将其设置为张量，因此将一直保持不变
        self.delta = 0.6 # 参考原实现的超参数
        self.lambda_c = 3.0 if data_name=="cifar100" else 8.0
        self.lambda_f = 1.0 if data_name=="cifar100" else 10.0
        pass

    

    def PODloss(self, true_classes, y_hat_c): 
        # lambda_c和lambda_f是超参数，论文里有给出推荐值
        # 池化蒸馏损失分为两个部分：空间损失和平坦损失
        # 先计算空间损失
        # 这里的stages包含了卷积网络每个阶段的输出，再加上最后的输出(FC层)

        # 无论是对哪个维度做池化操作，都需要算出每个像素点误差的平方
        # 对第一个任务进行训练的时候，没有L_spatial, 只有L_LSC

        # TODO: 完成NCALoss的部分
        L_LSC = utils.NCAloss(self.eta, self.delta, true_classes, y_hat_c, self.curModel.class_lst)

        if self.oldModel is None: #刚开始训练，还没有旧模型的情况，就只使用L_LSC
            return L_LSC
        else: 
            stages = self.curModel.stages
            oldStages = self.oldModel.stages
            assert(len(stages) == len(oldStages)), "需要保证前后两个模型backbone的阶段数一致"
            length = len(stages)

            # 这里的每个stages输入的形状估计是 batch_size X Channel X Height X width
            L_spatial = None

            for i in range(length-1):
                # 即池化后逐元素求差，再进行平方，再进行求和
                L_Width = (torch.sub(stages[i].sum(dim = 3), oldStages[i].sum(dim = 3))**2).sum()
                L_Height = (torch.sub(stages[i].sum(dim = 2), oldStages[i].sum(dim = 2))**2).sum()
                if L_spatial is None:
                    L_spatial = L_Width + L_Height
                else:
                    L_spatial += L_Width + L_Height
                
            '''
            QUOTE:
            For POD-spatial, before sum-pooling we take the features to the power of 2
                element-wise. The vector resulting from the pooling is then L2 normalized.
            '''
            # 没理解上面这段话
            
            

            # 这里要确保h的形状是 batch_size X length_
            L_flat = (torch.sub(stages[length-1], oldStages[length-1])**2).sum(dim=1)**0.5
            L_POD = self.lambda_c/(length-1) * L_spatial + self.lambda_f*L_flat
            L_POD = torch.mean(L_POD) # 需要对整个批次的损失做一个平均

            '''
            QUOTE:
            Following Hou et al. [14], we multiply both losses by
            the adaptive scaling factor: \lambda = \sqrt{N/T} with N being the number of seen classes
            and T the number of classes in the current task.
            '''
            # TODO: 完成上面两段引用所代表的操作

        return L_POD + L_LSC




class PODNet(nn.Module):
    def __init__(self, pretrainedBackbone = False, backboneType = 'resnet50'):
        super(PODNet, self).__init__()
        self.backbone = ResNet.construct_backbone(pretrainedBackbone, backboneType)
        if torch.cuda.is_available():
            self.backbone = self.backbone.to("cuda")
        self.fc = nn.Linear(512,128)
        self.proxyNums  = 10 

        self.stages = None #每种resnet的stage数量一致吗？
        self.output = None #存储每一次的输出
        
        self.proxys = None #存储每个类别的代理向量
        self.class_lst = None

    def forward(self, input):
        # 这里只输出h，分类交给LSC
        self.stages, self.output = self.backbone(input)
        batch_size = self.output.shape[0]
        self.output = self.fc(self.output.squeeze())
        self.output = torch.relu(self.output) #试试在最后加一个relu层
        return self.output.view(batch_size,-1,1,1)
    
    def LSC_Classifier(self, output_h, predict_only = True):
        # 这里对输出的h与每个类别的K个proxy计算相似度，并将得分最高的类别作为预测类别
        # 那么这里的每个类别的proxy有没有必要
        # 获取所有代理之后，对其进行计算
        # 假设proxy的数据类型是dict<int, tensor>
        # 这里返回的是预测类别，以及该类别经LSC运算后的y值(便于损失函数计算)

        # 这里的公式到底是什么意思？
        max_y_c = None
        predict_class = None
        class_lst = []
        y_c_lst = []
        # proxys = tensor(K, features)
        # output_h = tensor(batch_size, features)
        # 先把output_h squeeze一下，压缩掉原本的"1"维度，之后再在最后造出来一个"1"维度，强制使其形状一致
        output_h = torch.squeeze(output_h)
        output_h = output_h.unsqueeze(-1)   
        # output_h = tensor(batch_size, features, 1)
           
        for class_, proxys in self.proxys.items():

            # 这里直接赋值得了, 否则会产生原地操作(或者更好的方法应该是用no_grad()包裹适当的部分)
            scaler = torch.zeros((output_h.shape[0],1,1),dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
            y_c = torch.zeros((output_h.shape[0],1,1),dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")

            for i in range(self.proxyNums):
                item = utils.cosineSim(output_h, proxys[i])
                # 这里的输出应该是(batch, 1, 1)
                scaler += torch.exp(item)
                y_c += torch.exp(item)*item
            # 把每个类别按顺序记录下来
            class_lst.append(class_)
            # 这里y_c相当于对类别与所有代理相似度的一个加权平均
            y_c /= scaler
            #比较所有y_c，选择最大的作为预测类别
            # TODO: 注意这里的y_c是批量的数据，因此不能如此粗暴地得到最大的y_c以及对于分类 √
            # 这里要首先确保y_c 的形状是(batchsize, 1) --> 没有必要
            # torch.squeeze(y_c)
            if max_y_c is None:
                max_y_c = y_c
                predict_class = torch.full(y_c.shape, class_)
            else:
                predict_class[y_c>max_y_c] = class_
                max_y_c[y_c>max_y_c] = y_c[y_c>max_y_c]


            # 这里的y_c_lst 包含了所有的y_hat_c, 可以理解为对每个类别c的预测概率
            # 确保y_c被压缩为b, 之后就可以用torch.stack(y_c_lst,axis = -1)将其形状变为(batch,class)
            y_c_lst.append(y_c.squeeze())
        
        # 注意: 这里y_c_lst的形状硬要说的话应该是(class, batch), 后续在计算的时候应该把形状变换为(batch, class)
        # predict_class的形状应该是(batch, 1)
        predict_class = predict_class.view(-1,1)
        assert(max_y_c is not None), "Unexpected Error: LSC无法进行分类, 可能是由传导数据错误引发的"
        if predict_only:
            return max_y_c, predict_class
        else:
            # return predict_class, y_c_lst
            self.class_lst = class_lst
            return y_c_lst
        
        # TODO: 区分NME策略和CNN策略 -24.5.31 X
        # 不需要区分了


    def append_proxys(self, new_proxys):
        # 这里要确保两个字典里的张量在同一设备上(尽可能在gpu上)
        if self.proxys is None:
            # 说明是首次训练，直接将其进行赋值即可  
            self.proxys = new_proxys
        else:
            self.proxys.update(new_proxys)


# 真正的推理任务中，分类使用了NME(icarl的最近邻策略)或者CNN(UCIR提出的一种新策略)
