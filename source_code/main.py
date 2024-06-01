# import torch
# from torch import nn


# 首先还是要想清楚模型的架构
# 应该分出一个模块来写model
# 其结构类似下面的图解: 
# model-----
#          |--O __init__.py
#          |--O ResNet.py
#          |--O PODNet.py
#          |--O utils.py
# 还应该有一个模块来写训练的内容
# train------
#           |--O __init__.py
#           |--O train.py
# 一个模块来写测试的内容，metrics的相关内容也写在这里
# test-------
#           |--O __init__.py
#           |--O test.py
#           |--O metric.py
# 一个模块写数据处理的内容
# dataprocess--------
#                   |--O __init__.py
#                   |--O process.py
# main函数要写好接口与超参数设置
# 因为算力限制，还需要检查gpu是否可用，同时可以考虑用分布式的训练方式


from test import test
import argparse
test.testfunc()
test.showcwd()
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataName", type = str, default="cifar100",
#                         help="dataset used in training or testing task, options: cifar100, imagenet100")
#     parser.add_argument("--task", type = str, default="train" ,
#                         help="task to be executed, options: train, test, inference")
#     parser.add_argument("--batchSize", type = int, default=64,
#                         help="batchsize of the model")
#     parser.add_argument("--loadPretrianed", type = bool, default=True,
#                         help="load the pretrained INCREMENTAL model or not, options: Ture, False")
#     parser.add_argument("--pretrainedPath", type = str, default = None,
#                         help="the path to the selected pretrained INCREMENTAL model")
    #etc
    