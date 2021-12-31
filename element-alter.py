# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 14:16:55 2019

@author: hhy
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt

#加载数据 #composition-TCP121032 composition-parameters143951 composition-gamaprime144000
data = np.loadtxt(r'E:\科研\机器学习\所做工作\composition-parameters20190125\composition-parameters143951.txt',delimiter='\t') 
Predict = np.loadtxt('C:/Users/Uaena_HY/Desktop/备选成分2021.3.15/153热物性参数.txt',delimiter='\t')
#数据集打乱
data=np.random.permutation(data)
x = []
y = []
for line in range(143951):   #composition-TCP121032 composition-parameters143951 composition-gamaprime144000
    x.append(data[line][:9])
    y.append(data[line][-2]) #143951时，-1:γ'析出温度  -2：固相线  -3：液相线
x = np.array(x)
y = np.array(y)

#划分数据集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.1,random_state=1)

#L2正则化
ridge = Ridge().fit(x_train,y_train)
print('ridge training set score:{:.2f}'.format(ridge.score(x_train,y_train)))
print('ridge testing set score:{:.2f}'.format(ridge.score(x_test,y_test)))
print("ridge.coef_: {}".format(ridge.coef_))
print("ridge.intercept_: {}".format(ridge.intercept_))
print("\n")
Ridge_pred = ridge.predict(Predict) 

#可视化
#plt.rcParams['font.sans-serif'] = ['SimHei'] #可以显示中文
#plt.rcParams['axes.unicode_minus'] = False #可以显示负号
#plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],Ridge_pred,color='Red', lw=2, label='predict')
#plt.xlabel('Co(wt.%)',fontsize = 15) 
#plt.ylabel('  Precipitation temperature of γ′ ',fontsize = 15)  #The content of TCP   Liquidus   Solidus  Precipitation temperature of gamma prime
#plt.title('Re元素对液相线的影响',fontsize = 20)  #Ru元素对液相线的影响  Ru对TCP相的影响  'Ru元素对r'+'’'+'析出温度的影响'
#plt.tick_params(labelsize = 14)
#plt.legend()  
#plt.show()