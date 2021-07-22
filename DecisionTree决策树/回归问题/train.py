from sklearn import tree #导入需要的模块
import numpy as np
from sklearn.metrics import accuracy_score
train_data=np.load('train_data.npz',allow_pickle=True)#加载数据
test_data=np.load('test_data.npz',allow_pickle=True)

train_x=train_data['arr_0'][:,:-1]#训练集
train_y=train_data['arr_0'][:,-1]#训练集标签

test_x=test_data['arr_0'][:,:-1]
test_y=test_data['arr_0'][:,-1]


model = tree.DecisionTreeClassifier(criterion='entropy')#加载分类决策树模型,这里用信息熵entropy作为信息复杂度度量，也可以用基尼系数'gini'

model.fit(train_x,train_y) #用训练集数据训练模型

predict = model.predict(test_x)#预测

print("accuracy_score: %.4lf" % accuracy_score(predict,test_y))
