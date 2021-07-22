import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

data=pd.read_csv('weather_data.txt',header=None).values#加载数据
data=data.reshape([1,len(data)])[0]

train_x = []
train_y = []
test_x = []
test_y = []

for i in range(int(len(data)*0.8)):
    train_x.append(list(data[i:i+5]))
    train_y.append(data[i+5])

for i in range(int(len(data)*0.8),len(data)-5):
    test_x.append(list(data[i:i+5]))
    test_y.append(data[i+5])

train_x=np.array(train_x)#将列表转换为数组
train_y=np.array(train_y)
test_x=np.array(test_x)
test_y=np.array(test_y)

print('train shape=',train_x.shape)
print('test shape=',test_x.shape)

model = XGBRegressor()#加载随机森林回归模型
model.fit(train_x,train_y)
predict = model.predict(test_x)

print('XGB回归 mean_absolute_error: %.4lf' % mean_absolute_error(predict,test_y))