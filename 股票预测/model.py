import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
plt.rcParams['font.sans-serif']=['SimHei']#显示中文
plt.rcParams['axes.unicode_minus']=False#显示负号

#Hyperparams
batch_size = 128
lr = 1e-6
epochs = 600
num_neurons = [32,32,64,64,128,128]
kp = 1.0

#定义输出层的weight和bias
w = tf.Variable(tf.random_normal([num_neurons[-1],1]))
b = tf.Variable(tf.random_normal([1]))

def load_data():
    train_x_batch = np.load(r'train_x_batch.npy',allow_pickle=True)
    train_y_batch = np.load(r'train_y_batch.npy',allow_pickle=True)
    return (train_x_batch,train_y_batch)

#定义lstm单元
def lstm_cell(units,keep_prob):
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=units,forget_bias=0.9)#activation默认为tanh
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

#定义lstm网络
def lstm_net(x,w,b,num_neurons,keep_prob):
    #将输入变成一个列表，列表的长度及时间步数
    inputs = tf.unstack(x,8,1)
    cells = [lstm_cell(units=n,keep_prob = keep_prob) for n in num_neurons]
    stacked_lstm_cells = tf.contrib.rnn.MultiRNNCell(cells)
    outputs,_ =  tf.contrib.rnn.static_rnn(stacked_lstm_cells,inputs,dtype=tf.float32)
    return tf.matmul(outputs[-1],w) + b
    

if __name__ == '__main__':

    #载入数据
    (train_x,train_y) = load_data()
    
    #定义placeholder
    x = tf.placeholder(shape=(None,8,8),dtype=tf.float32)
    y = tf.placeholder(shape=(None,1),dtype=tf.float32)
    keep_prob = tf.placeholder(tf.float32,[])

    #定义预测函数、损失函数、优化函数、初始函数、保存函数
    pred = lstm_net(x,w,b,num_neurons,keep_prob)
    
    cost = tf.reduce_mean(tf.reshape(tf.pow((pred-y),2),[-1]))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cost)
    init  = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())
    
    #启动交互式Session
    sess = tf.InteractiveSession()

    #训练模型
    sess.run(init)
    losses = []#记录每个epoch结束时的损失值
    for epoch in range(epochs):
        for step in range(train_x.shape[0]//batch_size+1):
            batch_x = train_x[step*batch_size:(step+1)*batch_size]
            batch_y = train_y[step*batch_size:(step+1)*batch_size]
            sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:kp})
            
        loss = sess.run(cost,feed_dict={x:batch_x,y:batch_y,keep_prob:1.0})
        losses.append(loss)
        print('Epoch[{}/{}]，Loss = {:.4f}\n'.format(epoch+1,epochs,loss))

    #可视化训练过程
    plt.plot(losses)
    plt.ylim(0,6)
    plt.title('损失值随迭代周期的改变')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.show()
        
    #保存模型
    #saver.save(sess,r'model_data_4\my_model.ckpt')
    
    #关闭会话
    sess.close()
    

        

    


    
