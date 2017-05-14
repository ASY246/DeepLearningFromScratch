# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 22:16:19 2016

自己写的一个简化版word2vec
@author: ASY
"""

import numpy as np
import NNSelfFunction as selfFuc

def normalizeRows(x):
    #传入列向量，按行归一化
    N = x.shape[0]
    x = x/np.sqrt(np.sum(x**2,axis = 1)).reshape(N,1) + 1e-20  #前面做完是一维的，因此reshape不能省略
    return x
    
def softmaxCostAndGradient(predicted, target, outputMatrixs):#target表示上下文context词的index,predicted表示
    #根据当前的输入输出权重矩阵状态，返回损失函数和梯度
    probabilities = selfFuc.softmax(predicted.dot(outputMatrixs.T))#输入和输出矩阵相乘
    cost = -np.log(probabilities[target])#只需要计算对应target的交叉熵，对应的向量维度为1，其他为0，不要
    
    delta = probabilities #delta用于计算梯度，记着交叉熵求导的公式
    delta[target] -= 1 #y尖减y
    
    N = delta.shape[0]
    D = predicted.shape[0]

    #word2vec的公式推导而来，分别对应两个梯度
    grad = delta.reshape((N,1))*predicted.reshape((1,D))
    gradPred = (delta.reshape((1,N)).dot(outputMatrixs)).flatten()

    return cost,gradPred,grad

#def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K = 10):    
    
def skipgram(currentWord, contextWords, tokens, inputMatrixs, outputMatrixs, 
             word2vecCostAndGradient = softmaxCostAndGradient):
    #根据当前词，当前上下文，词典，输入向量矩阵，输出向量矩阵，返回利用skipgram方法加和的gradIn和gradOut
    currentI = tokens[currentWord]  #输入词的index

    predicted = inputMatrixs[currentI, :] #从inputMatrix中把输入词对应的行拿出来，即为输入词对应的初始化词向量

    cost = 0.0
    gradIn = np.zeros(inputMatrixs.shape)#初始化输入权值矩阵梯度，对于skip-gram，只有一列有效
    gradOut = np.zeros(outputMatrixs.shape)#初始化输出权值矩阵梯度
    for cwd in contextWords: #将预测词对于每个context的词的交叉熵加和
        #将所有的损失函数cost梯度加和
        idx = tokens[cwd]
        cc,gp,gg = word2vecCostAndGradient(predicted, idx, outputMatrixs)
        cost += cc
        gradOut += gg
        gradIn[currentI, :] += gp

    #对于一个神经网络中有多个梯度需要求值，对应sgd的简便做法，就是把所有层的梯度拼成一个矩阵，然后把对应各自的梯度拼成一个矩阵
    #随后直接更新这个矩阵，就等于处理了网络中所有的梯度下降权值
    
    gradAll = np.concatenate((gradOut, gradIn),axis = 0)
    
    return cost, gradAll

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset, K = 10):
    #K = 10表示取用负例的数量
    grad = np.zeros(outputVectors.shape)
    gradPred = np.zeros(predicted.shape)
    
#    indices = [target]  #正例词的index
#    for k in range(K):  #负例采样K个样本
#        newidx = dataset.sampleTokenIdx()
#        while newidx == target:
#            newidx = dataset.sampleTokenIdx()
#        indices += [newidx]
#
#    labels = np.array([1] + [-1 for k in range(K)]) #采样后样本标签的集合
#    vecs = outputVectors[indices, :]    #取出对应向量的输出矩阵
#
#    t = selfFuc.sigmoid(vecs.dot(predicted) * labels) #来自note上的公式推导
#    
#    cost = -np.sum(np.log(t))   #note4公式推导
#    
#    delta = labels*(t-1)
#    
#    gradPred = delta.reshape((1, K+1)).dot(vecs).flatten()
#    gradtemp = delta.reshape((K+1,1)).dot(predicted.reshape(1,predicted.shape[0]))
#    
#    for k in range(K + 1):
#        grad[indices[k]] += gradtemp[k, :]
    
    t = selfFuc.sigmoid(predicted.dot(outputVectors[target,:]))
    cost = -np.log(t) #计算正例样本的cost function，正常做
    delta = t - 1  # y尖-y

    gradPred += delta * outputVectors[target, :]  #计算交叉熵对于正例样本的两个梯度
    grad[target, :] += delta * predicted
    
    for k in range(K):   #对应计算负例样本的cost和grad
        idx = dataset.sampleTokenIdx()
        
        t = selfFuc.sigmoid(-predicted.dot(outputVectors[idx,:]))  #为了使负例采样的样本以更小的概率被取到，因此这里面加一个负号
        
        cost += -np.log(t)  #按照note4的公式计算
         
        delta = 1 - t   #因为负例的缘故，t和label的负号都相反了

        gradPred += delta * outputVectors[idx, :]
        grad[idx, :] += delta * predicted

    return cost, gradPred, grad
    
#    
#def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors, 
#         dataset, word2vecCostAndGradient = softmax):
#def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
#    # 加一个wrapper函数，用mini-batch的方法从训练集中取样作训练，在简单word2vec中不用
#    batchsize = 50
#    cost = 0.0
#    grad = np.zeros(wordVectors.shape)
#    N = wordVectors.shape[0]
#    inputVectors = wordVectors[:N/2,:]
#    outputVectors = wordVectors[N/2:,:]
#    for i in range(batchsize):
#        C1 = random.randint(1,C)
#        centerword, context = dataset.getRandomContext(C1)
#        
#        if word2vecModel == skipgram:
#            denom = 1
#        else:
#            denom = 1
#        
#        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
#        cost += c / batchsize / denom
#        grad[:N/2, :] += gin / batchsize / denom
#        grad[N/2:, :] += gout / batchsize / denom
#        
#    return cost, grad
def test_word2vec():
    #测试数据
    #dataset = type('dummy', (), {})() #dataset用于负例采样
    initialMat = normalizeRows(np.random.randn(10,3))#随机初始化权值矩阵，前五个为inputvector初始化，后五个为outputvector初始化
    tokens = dict([("a",0),("b",1),("c",2),("d",3),("e",4)])#词典
    print("\nthe result:\n")
    #print(skipgram("c",["a","b","e","d","b","c"], tokens, initialMat[:5,:], initialMat[5:,:]))#dataset为负例采样数据集
    wordVectors = np.concatenate(((np.random.rand(5, 3) - .5)/3, np.zeros((5, 3))), axis=0)   # 给wordvector设初始值，5个词，3维向量,输入矩阵随机初始化（-0.5,0.5），输出矩阵0初始化
    
    #每次都使用这一个文本窗来做
    wordVectors0 = selfFuc.sgd(skipgram("c",["a","b","e","d","b","c"],tokens,initialMat[:5,:],
                                        initialMat[5:,:]), wordVectors, 0.3, 40000)  #优化函数，初始值，步长，迭代次数

    
    #    cbow("a", 2, ["a", "b", "a", "c"], tokens, initialMat[:5,:], initialMat[5:,:], dataset, negSamplingCostAndGradient)
#    
    wordVectors = (wordVectors0[:5,:] + wordVectors0[5:,:])
    print(wordVectors)


if __name__=="__main__":
    test_word2vec()