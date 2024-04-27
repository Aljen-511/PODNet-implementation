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
# 一个模块写数据处理的内容
# dataprocess--------
#                   |--O __init__.py
#                   |--O dataprocess.py
# main函数要写好接口与超参数设置


from test import test

test.testfunc()

