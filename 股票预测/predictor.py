import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.rcParams['axes.unicode_minus']=False#显示负号

def load_data():
    test_x_batch = np.load(r'test_x_batch.npy',allow_pickle=True)
    test_y_batch = np.load(r'test_y_batch.npy',allow_pickle=True)
    return (test_x_batch,test_y_batch)

#定义lstm单元
def lstm_cell(units):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=units,forget_bias=0.0)#activation默认为tanh
    return cell

#定义lstm网络
def lstm_net(x,w,b,num_neurons):
    #将输入变成一个列表，列表的长度及时间步数
    inputs = tf.unstack(x,8,1)
    cells = [lstm_cell(units=n) for n in num_neurons]
    stacked_lstm_cells = tf.contrib.rnn.MultiRNNCell(cells)
    outputs,_ =  tf.contrib.rnn.static_rnn(stacked_lstm_cells,inputs,dtype=tf.float32)
    return tf.matmul(outputs[-1],w) + b

#超参数
num_neurons = [32,32,64,64,128,128]

#定义输出层的weight和bias
w = tf.Variable(tf.random_normal([num_neurons[-1],1]))
b = tf.Variable(tf.random_normal([1]))

#定义placeholder
x = tf.placeholder(shape=(None,8,8),dtype=tf.float32)

#定义pred和saver
pred = lstm_net(x,w,b,num_neurons)
saver = tf.train.Saver(tf.global_variables())

if __name__ == '__main__':
    
    #开启交互式Session
    sess = tf.InteractiveSession()
    saver.restore(sess,r'D:\股票预测\model_data_4\my_model.ckpt')

    #载入数据
    test_x,test_y = load_data()

    #预测
    predicts = sess.run(pred,feed_dict={x:test_x})
    predicts = ((predicts.max() - predicts) / (predicts.max() - predicts.min()))#数学校准

    #可视化
    plt.plot(predicts,'r',label='预测曲线')
    plt.plot(test_y,'g',label='真实曲线')
    plt.xlabel('第几天/days')
    plt.ylabel('开盘价(归一化)')
    plt.title('股票开盘价曲线预测(测试集)')
    plt.legend()
    
    plt.show()

    #关闭会话
    sess.close()

    








              
