import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#读取训练数据
train_data = pd.read_excel('train_data.xlsx',index_col=0)
train_arrs = np.array(train_data.iloc[:,:])
train_xs = train_arrs[:,-8:].astype('float32')
train_ys = (np.array(train_data['open'],dtype='float32')).reshape(-1,1)
#读取测试数据
test_data = pd.read_excel('test_data.xlsx',index_col=0)
test_arrs = np.array(test_data.iloc[:,:])
test_xs = test_arrs[:,-8:].astype('float32')
test_ys = (np.array(test_data['open'],dtype='float32')).reshape(-1,1)

#归一化
train_ys = (train_ys-train_ys.min()) / (train_ys.max() - train_ys.min())
test_ys  = (test_ys-test_ys.min())   / (test_ys.max()  - test_ys.min())
for dim in range(train_xs.shape[1]):
    train_xs[:,dim] = (train_xs[:,dim] - train_xs[:,dim].min()) / (train_xs[:,dim].max() - train_xs[:,dim].min())
	
for dim in range(test_xs.shape[1]):
    test_xs[:,dim]  = (test_xs[:,dim] - test_xs[:,dim].min()) / (test_xs[:,dim].max() - test_xs[:,dim].min())

#由于是预测任务，那么数据的第一个维度会少掉一个time_step-1
time_step = 8
input_dim = 8

aranged_train_xs = np.zeros(shape=(train_xs.shape[0]-time_step+1,time_step,input_dim))
for idx in range(aranged_train_xs.shape[0]):
    aranged_train_xs[idx] = train_xs[idx:idx+8]
    
aranged_test_xs = np.zeros(shape=(test_xs.shape[0]-time_step+1,time_step,input_dim))
for idx in range(aranged_test_xs.shape[0]):
    aranged_test_xs[idx] = test_xs[idx:idx+8]
    
aranged_train_ys = train_ys[time_step-1:]
aranged_test_ys  =  test_ys[time_step-1:]

#保存数据
np.save(r'train_x_batch.npy',aranged_train_xs)
np.save(r'train_y_batch.npy',aranged_train_ys)
np.save(r'test_x_batch.npy',aranged_test_xs)
np.save(r'test_y_batch.npy',aranged_test_ys)





