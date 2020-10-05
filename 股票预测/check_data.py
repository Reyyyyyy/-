import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#check train data and test data
train_data = pd.read_excel('train_data.xlsx',index_col=0)
train_arrs = np.array(train_data.iloc[:,:])
trains = train_arrs[:,-9:].astype('float32')


test_data = pd.read_excel('test_data.xlsx',index_col=0)
test_arrs = np.array(test_data.iloc[:,:])
tests = test_arrs[:,-9:].astype('float32')

#normalize
for dim in range(trains.shape[1]):
    trains[:,dim] = (trains[:,dim] - trains[:,dim].min()) / (trains[:,dim].max() - trains[:,dim].min())
	
for dim in range(tests.shape[1]):
    tests[:,dim]  = (tests[:,dim] - tests[:,dim].min()) / (tests[:,dim].max() - tests[:,dim].min())
    
#visualization of train data
for dim in range(trains.shape[1]):
    plt.subplot(3,3,dim+1)
    plt.plot(trains[:,dim])
    plt.title('dim %d'%(dim+1))

plt.show()

#visualization of test data
for dim in range(tests.shape[1]):
    plt.subplot(3,3,dim+1)
    plt.plot(tests[:,dim])
    plt.title('dim %d'%(dim+1))

plt.show()
