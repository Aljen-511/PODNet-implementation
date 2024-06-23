import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, ".."))
from dataprocess.process import generalDataSet
from model.PODNet import integratedMdl
from torch.utils.data import DataLoader
import torch
import numpy as np
from tqdm import tqdm

def testfunc():
    print("just test a mode")
    print(os.path.dirname(os.path.abspath(__file__)))

def showcwd():
    print(os.getcwd())


def acc_loop(
        model: integratedMdl,
        baseTestset: generalDataSet,
        inc_step: int
):
    '''
    model: 训练完的模型
    baseTestset: 测试数据集, generalDataSet类型
    inc_step: 增量学习的类别数量
    '''
    # TODO: 完成测试的基本循环单元，用以返回准确率 6.2 √
    num_workers_ = 2
    batch_size_ = 64

    basic_test_set = baseTestset.getIncData4Test(inc_num=inc_step)
    test_dataloader = DataLoader(basic_test_set, shuffle=False, num_workers=num_workers_, batch_size=batch_size_)
    true_predict = 0
    totol_predict = 0
    with torch.no_grad():
        # for images, labels in test_dataloader:
        for images, labels in tqdm(test_dataloader, desc=f"testing...",unit="itr",leave=False):
            images = images.to("cuda" if torch.cuda.is_available() else "cpu")
            # 这里仿照basic_loop里面的做法
            y_output = model.curModel(images)
            _, result = model.curModel.LSC_Classifier(y_output)
            result = result.to("cpu").numpy()
            labels = labels.view(-1,1).numpy()
            true_predict += np.sum(result == labels)
            totol_predict += result.shape[0]

    return true_predict/totol_predict
