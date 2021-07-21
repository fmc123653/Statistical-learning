import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,classification_report
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB
train_data=np.load('train_data.npz',allow_pickle=True)#加载数据
test_data=np.load('test_data.npz',allow_pickle=True)

train_x=train_data['arr_0'][:,:-1]#训练集
train_y=train_data['arr_0'][:,-1]#训练集标签

test_x=test_data['arr_0'][:,:-1]
test_y=test_data['arr_0'][:,-1]



Mnb = MultinomialNB()#加载多项式朴素贝叶斯模型
Bnm = BernoulliNB()#加载伯努利朴素贝叶斯模型
Gnb = GaussianNB()#加载高斯朴素贝叶斯模型


Mnb.fit(train_x,train_y)
Mpredict = Mnb.predict(test_x)

Bnm.fit(train_x,train_y)
Bpredict = Bnm.predict(test_x)

Gnb.fit(train_x,train_y)
Gpredict = Gnb.predict(test_x)

print("多项式模型 accuracy_score: %.4lf" % accuracy_score(Mpredict,test_y))
print("伯努利模型 accuracy_score: %.4lf" % accuracy_score(Bpredict,test_y))
print("高斯模型 accuracy_score: %.4lf" % accuracy_score(Gpredict,test_y))
