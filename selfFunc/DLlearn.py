# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 10:21:45 2016

kaggle上面树叶分类的竞赛，使用softmax完成
整体的数据传输思路就是用传统的readlines把数据集传输到列表中，然后列表转换为矩阵np.matrix，再传入到tensorflow做运算

@author: ASY
"""

#import tensorflow as tf
#
#filename_queue = tf.train.string_input_producer(["D:\Competition\Kaggle\LeafClassify\train.csv"])
#
#reader = tf.TextLineReader()
#key, value = reader.read(filename_queue)
##设置列的默认值
#record_defaults = [[0] for item in range(194)]
#columns = tf.decode_csv(value, record_defaults = record_defaults)
#features = tf.pack(columns[3:])
#
#with tf.Session() as sess:
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    
#    for i in range(1200):
#        example, label = sess.run([features, columns[2]])
#
#    coord.request_stop()
#    coord.join(threads)

import pandas as pd
import os

trainFile = 'D:/Competition/Kaggle/LeafClassify/train.csv/train.csv'

if os.path.exists('D:/Competition/Kaggle/LeafClassify/train.csv/trainWithoutName.csv') == 0:
    train_df = pd.read_csv(trainFile)#df的默认读入数据的方法把trainFile的列名自动读入进去，如果需要自己设置可以header= None, names =等
    train_df.to_csv('D:/Competition/Kaggle/LeafClassify/train.csv/trainWithoutName.csv', index = False, header = False)
#下面使用原始方法读入数据
with open('D:/Competition/Kaggle/LeafClassify/train.csv/trainWithoutName.csv') as trainFile:
    rawTrain = trainFile.readlines()
print('the amount of trainSamples:', len(rawTrain))

labelSet = set()
for line in rawTrain:
    labelSet.add(line.split(',')[1])

labelSet = sorted(list(set(labelSet)))
print('the amount of labelSet', len(labelSet))

label_indices = dict((c, i) for i, c in enumerate(labelSet))
indices_char = dict((i, c) for i, c in enumerate(labelSet))

trainFeature = []
trainLabel_list = []
for line in rawTrain:
    lineList = []
    for index in range(len(line.split(','))):
        if index == 1:#文件中第二列为类别
            trainLabel_list.append(int(label_indices[line.split(',')[1]]))
        elif index >= 2:
             lineList.append(float(line.split(',')[index]))
            
    trainFeature.append(lineList)
    
#把label转化为one-hot向量
#这里把trainFeature和trainLabel转换为np.ndarrays才能矩阵乘，那么如何把外部的数据转换为矩阵乘呢？？？？
import numpy as np
trainFeature = np.mat(trainFeature)
#trainLabel_list = np.mat(trainLabel_list).I
trainLabel = np.zeros([990,99])
for index in range(len(trainLabel)):
    trainLabel[index][trainLabel_list[index]]=1
#matrix是ndarray的子类，所以前面ndarray的优点都保留，另外matrix全部都是二维的，并且加入了一些更符合直觉的函数
#mat.I表示逆矩阵，乘法表示的是矩阵相乘的结果

#上tensorflow

import tensorflow as tf

#把在外面弄好的数据传入到tf的数据流中


x = tf.placeholder(tf.float32,[None, 192])#向量维数为192
#初始化变量为0
W = tf.Variable(tf.zeros([192, 99]))
b = tf.Variable(tf.zeros([99]))#要生成的one-hot向量为99维

#激活函数

y = tf.nn.softmax(tf.matmul(x,W) + b)#继续建图


y_correct = tf.placeholder(tf.float32, [None, 99])#建立y_correct
#交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_correct * tf.log(y), reduction_indices=[1]))
#softmax是凸函数使用全局梯度下降
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

init = tf.global_variables_initializer()

x_test = tf.placeholder(tf.float32,[None, 192])#测试数据特征输入
#导入测试数据
with open('D:/Competition/Kaggle/LeafClassify/test.csv/test.csv') as testFile:
    rawTest = testFile.readlines()
testFeature = []

for line in rawTest:
    lineList = []
    for index in range(len(line.split(','))):
        if index >= 1:
            lineList.append(float(line.split(',')[index]))
    testFeature.append(lineList)
    
testFeature = np.mat(testFeature)



with tf.Session() as sess:
    sess.run(init)
    for epoch in range(2000):#迭代1000次            
        _, CE_res = sess.run([optimizer,cross_entropy], feed_dict={x:trainFeature, y_correct:trainLabel})#feed的数据为python的原本数据，不是tf转化的常量
        if epoch% 100 ==0:
            print("epoch=" , epoch, "CE_res=", CE_res)         
    print("Optimization Finished\n")
    print("CE_res=",CE_res,'\n')
    print("W=",sess.run(W),'\n',"b=",sess.run(b),'\n')
    
    #输出训练集精度
    correct_prediction = tf.equal(tf.argmax(y_correct,1), tf.argmax(y,1))
    #这个计算方法
    accuracy_train = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy_train",accuracy_train.eval({x:trainFeature,y_correct:trainLabel}))
    
    #输出测试结果
    
    y_test = tf.nn.softmax(tf.matmul(x,W) + b)
    
    #这里面使用什么方式把tf的object拿出来,eval
    y_test = y_test.eval({x:testFeature})
    print("y_test", y_test)
    

test_res = np.zeros([594,99])
for index in range(len(y_test)):
    test_res[index][np.argmax(y_test[index])] = 1
    



    

























    

