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


import yaml
import argparse
from train import loop
import sys
import os
def Parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataName", type = str, default="cifar100",
                        help="dataset used in training or testing task, options: cifar100, imagenet100")
    parser.add_argument("--task", type = str, default="train" ,
                        help="task to be executed, options: train, test, inference")
    parser.add_argument("--batchSize", type = int, default=64,
                        help="batchsize of the model")
    parser.add_argument("--loadPretrianed", type = bool, default=True,
                        help="load the pretrained INCREMENTAL model or not, options: Ture, False")
    parser.add_argument("--pretrainedPath", type = str, default = None,
                        help="the path to the selected pretrained INCREMENTAL model")
    #etc
    return parser


if __name__ == "__main__":
    # parser = Parser()
    # options = parser.parse_args()
    # print(options.dataName)
    # 将必要的路径导入环境变量, 避免后面的脚本中某些模块无法导入
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(cur_dir)
    sys.path.append(os.path.join(cur_dir, "model"))
    cfg_path = os.path.join(cur_dir,"cfg")

    try:
        with open(os.path.join(cfg_path, "options.yaml"), "r") as file:
            config = yaml.safe_load(file)
        loop.train_and_evaluate(
            inc_step = config["inc_step"],
            data_name = config["dataset"],
            train_ratio = config["train_ratio"],
            batch_size = config["batch_size"],
            max_epoch = config["max_epoch"],
            decay_rate = config["decay_rate"],
            SGD_momentum = config["SGD_momentum"],
            T_max = config["T_max"],
            SGD_learning_rate = config["SGD_learning_rate"]

        )        
    except FileNotFoundError:
        print("尝试打开一个不存在的文件, 请检查是否已在cfg文件夹下放置配置文件")
        
    except KeyError:
        print("请检查配置文件是否包含了必需的所有选项")



    