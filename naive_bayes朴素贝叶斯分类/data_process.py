import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def rebulid_picture(data):#将数据均值化处理成0或1
    for i in range(28):
        for j in range(28):
            if data[i][j]>=50:
                data[i][j]=1
            else:
                data[i][j]=0
    return data.reshape([1,28*28])[0]#将数据格式转换为1x784

train_data=[]
test_data=[]
dic_num={}

data = np.load('mnist.npz')#加载数据


id=0
for val in tqdm(data['x_train']):
    lab=data['y_train'][id]
    res=list(rebulid_picture(val))
    res.append(lab)#把标签作为最后一列
    train_data.append(res)
    id+=1

id=0
for val in tqdm(data['x_test']):
    lab=data['y_test'][id]
    res=list(rebulid_picture(val))
    res.append(lab)#把标签作为最后一列
    test_data.append(res)
    id+=1

train_data=np.array(train_data)#转换为数组格式
test_data=np.array(test_data)

print('train_data.shape=',train_data.shape)
print('test_data.shape=',test_data.shape)
np.savez('train_data.npz',train_data)#保存为train_data.npz文件
np.savez('test_data.npz',test_data)#保存为test_data.npz文件

