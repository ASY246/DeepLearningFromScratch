# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 22:00:43 2016

@author: ASY

改写自karpathy的代码

学习下代码结构
"""

import numpy as np

# 第一块dataI/O，读入文件，结构清晰

data = open(r'C:\Users\ASY\Desktop\input.txt', 'r').read()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size,vocab_size))#这句可以转化为format了

char_to_ix = {ch:i for i, ch in enumerate(chars)}
ix_to_char = {i:ch for i, ch in enumerate(chars)}
              
#超参数初始化
hidden_size = 100  #隐藏层节点的个数
seq_length = 25  #RNN打开的序列长度，RNN不是不需要指定长度的，这里面就相当于是n-gram的窗口大小
learning_rate = 1e-1

#模型参数
Wxh = np.random.randn(hidden_size, vocab_size)*0.01  # 输入到隐藏层的权重
Whh = np.random.randn(hidden_size, hidden_size)*0.01  # 隐藏层到隐藏层的权重
Why = np.random.randn(vocab_size, hidden_size)*0.01  # 隐藏层到输出
bh = np.zeros([hidden_size,1])  # 计算隐藏层状态时添加的偏差
by = np.zeros((vocab_size, 1))  # 计算输出值时候的偏差

def lossFun(inputs, targets, hprev):
    """
    输入,targets表示
    hprev是隐藏层初始状态
    返回损失函数，梯度和最后hidden layer的状态
    """
    xs, hs, ys, ps = {}, {}, {}, {}
    hs[-1] = np.copy(hprev)
    loss = 0
    # 前向传播
    for t in range(len(inputs)):  # 按照所有时间步依次计算
        xs[t] = np.zeros((vocab_size,1))
        xs[t][inputs[t]] = 1    # one-hot编码
        hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # 求隐藏层状态
        # 这一步利用的上一个时间步的隐藏层状态
        ys[t] = np.dot(Why, hs[t]) + by # 输出，作为下一个词的概率输出
        ps[t] = np.exp(ys[t])/np.sum(np.exp(ys[t]))  # softmax交叉熵输出
        print(targets[t])
        print('get')
        print(ps[t])
        loss += -np.log(ps[t][targets[t],0])  # softmax,只取对应y=1位置的值即可
        
    # 反向计算梯度，梯度和原来的值维度相同
    dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh),np.zeros_like(Why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])
    
    for t in reversed(range(len(inputs))):#t从后向前排列写法
    #演示BPTT
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1 # 交叉熵对softmax中的y求导
        dWhy += np.dot(dy, hs[t].T) # 交叉
        dby += dy
        dh = np.dot(Why.T, dy) + dhnext # 反向传播到h，dhnext是hs[t-1]
        dhraw = (1 - hs[t]*hs[t])* dh # 反向传播
        dbh += dhraw
        dWxh += np.dot(dhraw, xs[t].T)
        dWhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(Whh.T, dhraw)
        
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:  #python写法！
        np.clip(dparam, -5, 5, out = dparam) # 梯度爬升
        
    return loss,dWxh,dWhh,dWhy,dbh,dby,hs[len(inputs)-1]#返回hs的最后一个状态


#def sample(h, seed_ix, n): #n为展开的所有时间步
#    """
#    从模型中采样出一个整数序列
#    h是记忆状态，seed_ix是第一个时间步的seed letter
#    """
#    x = np.zeros((vocab_size, 1))
#    x[seed_ix] = 1
#    ixes = []
#    for t in range(n):
#        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
#        y = np.dot(Why, h) + by
        
n, p = 0, 0   #python的初始化写法, n为迭代的次数,p为序列初始化的下标
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why) #初始化写法
mbh, mby = np.zeros_like(bh), np.zeros_like(by)
smooth_loss = -np.log(1.0/vocab_size)*seq_length # 交叉熵初始值

while True:
    # 比较输入序列是否到数据集末尾，如果到了，就从头再来
    if p + seq_length + 1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size,1)) # 重置RNN的隐藏层，这步还是有必要的
        p = 0 
    inputs= [char_to_ix[ch] for ch in data[p:p+seq_length]] # 惯用写法
    targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

    # 从模型中采样--？？？这个采样没看出来有什么用！
#    if n % 100 == 0:
#        sample_ix = sample(hprev, inputs[0], 200)
#        txt = ''.join(ix_to_char[ix] for ix in sample_ix)
#        print ('----\n %s \n -----' % (txt,))

    #模型运作一次，要把对应每个时间步的梯度都求出来，一起更新
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001 # 这个loss平滑的作用？？一次只更新部分交叉熵- -
    if n % 100 == 0: print ('iter %d, loss: %f' % (n, smooth_loss))
    
    # 参数更新
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                  [dWxh, dWhh, dWhy, dbh, dby],
                                  [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)
    #各种梯度下降优化算法
    p += seq_length
    n += 1





