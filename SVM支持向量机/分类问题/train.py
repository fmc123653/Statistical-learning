import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

train_data=np.load('train_data.npz',allow_pickle=True)#加载数据
test_data=np.load('test_data.npz',allow_pickle=True)

train_x=train_data['arr_0'][:,:-1]#训练集
train_y=train_data['arr_0'][:,-1]#训练集标签

test_x=test_data['arr_0'][:,:-1]
test_y=test_data['arr_0'][:,-1]


model = SVC()
print('start training .....')
model.fit(train_x,train_y)
predict = model.predict(test_x)
print("SVM accuracy_score: %.4lf" % accuracy_score(predict,test_y))

