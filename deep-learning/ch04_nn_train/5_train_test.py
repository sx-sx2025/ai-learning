import torch
from torch import nn,optim

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        # 定义一个却连接参
        self.linear = nn.Linear(5,3)
        # 手动初始化参数
        self.linear.weight.data = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5],

            ]
        ).mT
        self.linear.bias.data = torch.tensor([1.0,2.0,3.0])

    def forward(self, x):
        x = self.linear(x)
        return x

if __name__ == '__main__':
    #1.准备数据
    x = torch.tensor([1,2,3,4,5]).float()
    target = torch.tensor([0,0,1]).float()

    model = MyModel()
    print("weight:",model.linear.weight)
    print("bias:",model.linear.bias)

    output = model(x)
    print("output:",output)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, target)
    print("loss:",loss)

    loss.backward()
    print("weight's gradient:",model.linear.weight.grad)

    print('雁阵梯度=======')
    proba = torch.softmax(output,dim=-1)
    print("proba:",proba)

    #softmaxwithloss
    E = (proba - target).reshape(1,-1)

    #linear
    X = x.reshape(1,-1)
    grad_W = X.T @ E
    print("grad W:",grad_W)

    #6.SGD 更新参数
    optimizer = optim.SGD(model.parameters(),lr=0.1)
    optimizer.step()

    print("更新参数：",model.linear.weight)

