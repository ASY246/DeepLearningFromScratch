# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 19:37:09 2016

@author: ASY

深度学习自己实现功能函数
"""

#先做mini-batch再把特征和标签分离高效
def SelectBatch(batchSize, inputData):
#简单随机抽样，用于mini-batch操作,输入1.batchsize大小，2.原始数据（嵌套列表表示）
#输出一个随机抽样的batchData
    import random
    outputData = random.random.sample(inputData,batchSize)#利用random.random.sample函数进行简单随机抽样
    
    return outputData
#one-hot编码
def oneHotDecoding(dim,inputList):
#输入：1.one-hot编码维度，2.列表，输出：one-hot编码
    import numpy as np
    output = np.zeros([len(inputList),dim])
    for index in range(len(inputList)):
        output[index][inputList[index]] = 1 
    return output

def sigmoid(inputData):
#sigmoid函数，输入为np数组，输出为point-wise的np数组
    import numpy as np
    outputData = 1/(1 + np.exp(-inputData))
    return outputData
    
def sigmoid_grad(inputData):
#求sigmoid函数的梯度
    outputData = inputData*(1-inputData)
    return outputData
    
def softmax(inputData):
#softmax函数，输入为np数组,输出为softmax归一化之后的np数组
    import numpy as np
    sumExp = np.sum(np.exp(inputData))
    outputData = np.exp(inputData)/sumExp
    return outputData
    
def forward_backward_prop(data, labels, params, dimensions):#单隐层网络，输入，输出什么，cs224
    """
    Forward and backward propagation for a two-layer sigmoidal network
    
    Compute the forward propagation and for the cross entropy cost, and backward propagation for
    the gradients for all parameters.
    """
    import numpy as np
    #Unpack network parameters
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    
    W1 = np.reshape(params[ofs:ofs + Dx*H], (Dx, H))
    ofs += Dx*H
    b1 = np.reshape(params[ofs:ofs + H], (1,H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H*Dy], (H, Dy))
    ofs += H*Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    
    #forward propagation
    layer1_output = sigmoid(data.dot(W1)+b1)#np.array相乘使用.dot
    layer2_output = softmax(layer1_output.dot(W2)+b2)
    cost = -np.sum(labels.dot(np.log(layer2_output)))#交叉熵是一个数值，需要把向量的所有维度做一个加和
    
    #backward propagation----------------------写法记一下，分两组，delata变化，w不变化
    delta2 = layer2_output - labels
    gradW2 = delta2.dot(layer1_output)
    gradb2 = np.sum(delta2, axis = 0)
    
    delta1 = delta2.dot(W2.T)*sigmoid_grad(layer1_output)
    gradW1 = delta1.dot(data)
    gradb1 = np.sum(delta1, axis = 0)
    
    #Stack gradients
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),gradW2.flatten(),gradb2.flatten()))
    
    return cost, grad
    
    
def save_params(iter,params):
    #保存参数
    import cPickle as pickle
    import random
    with open("saved_params_%d.npy" % iter, "w") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)#这句是干嘛的？？
        
#def load_saved_params():
#    st
#保存参数

def sgd(f, x0, step, iterations):
    #随机梯度下降函数
    #
    #f为优化的函数，返回损失函数和梯度
    #x0为SGD初始的点
    #step为步长
    #iterations为迭代次数
    
    #Output：
    #x为SGD迭代结束后的参数值
    import numpy as np
    ANNEAL_EVERY = 20000  #每过20000步改变步长
    
    start_iter = 0
    x = x0
    N = x.shape[0]
    postprocessing = lambda x: x/(np.sum(x,axis = 1).reshape(N,1) + 1e-10)
    
    for iter in range(start_iter + 1, iterations + 1):   #这个+1的写法可以学学。。。
        
        cost = None
        cost,grad = f #获取损失函数和梯度
        x -= step * grad  #梯度下降

        #x = x/(np.sum(x,axis = 1).reshape(N,1) + 1e-10) #归一化处理
        x = postprocessing(x)
        
    if iter % ANNEAL_EVERY == 0:
        step *= 0.5
        
    return x
          
    
    
    
def tf_BGD():
    import tensorflow as tf
    import numpy as np
    
    #随机生成100点（x,y）
    x_data = np.random.rand(100).astype(np.float32)
    y_data = x_data*0.1 + 0.3
    
    #构建线性模型的tensor变量W，b
    W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
    b = tf.Variable(tf.zeros([1]))
    y = W*x_data + b
    
    #构建损失方程，优化器及训练模型
    loss = tf.reduce_mean(tf.square(y - y_data))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    train = optimizer.minimize(loss)
    
    #构建变量初始化操作init
    init = tf.initialize_all_variables()
    
    #构建TensorFlow Session
    sess = tf.Session()
    
    #初始化所有TensorFlow变量
    sess.run(init)
    
    #训练该线性模型，每隔20次迭代，输出模型参数
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(W), sess.run(b))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    