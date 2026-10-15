import torch
from torch.optim import SGD
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 0.05 * x[0]**2 + x[1]**2

#定义梯度下降垫带函数，传入起始点，优化器，垫带次数，返回现将轨迹列表
def gradient_descent(x,optimizer,n_iters):
    x0_list = []
    x1_list = []
    #循环垫带
    for i in range(n_iters):
        x0_list.append(x[0].item())
        x1_list.append(x[1].item())
        #1前向传播,计算损失
        loss = f(x)
        #2反向传播，计算梯度
        loss.backward()
        #3更新参数
        optimizer.step()
        #4梯度清0
        optimizer.zero_grad()
    return x0_list,x1_list

#定以超参数
lr = 0.01
n_iters = 500

#初始化参数
x = torch.tensor([-7.0,2.0],requires_grad=True)

#sgd
optimizer = SGD([x],lr=lr,momentum=0.9)
x0_list,x1_list = gradient_descent(x,optimizer,n_iters)
plt.plot(x0_list,x1_list,color='red')


#动量法
x = torch.tensor([-7.0,2.0],requires_grad=True)

optimizer = SGD([x],lr=lr)

x0_list,x1_list = gradient_descent(x,optimizer,n_iters)
plt.plot(x0_list,x1_list,color='blue')
plt.show()



