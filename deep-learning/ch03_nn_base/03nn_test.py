import torch
import torch.nn as nn

#自定义神经网络模型
class MyModel(nn.Module):
    #初始化
    def __init__(self):
        super(MyModel, self).__init__()
        # 定义三个线性层
        self.linear1 = nn.Linear(3, 4)
        self.linear2 = nn.Linear(4, 4)
        self.out = nn.Linear(4, 2)
    #向前传播
    def forward(self, x):
        #第一层传递
        x = self.linear1(x)
        x = torch.tanh(x)
        # 第二层传递
        x = self.linear2(x)
        x = torch.relu(x)
        # 第三层传递
        x = self.out(x)
        y = torch.softmax(x,dim=-1)
        return y

#测试
if __name__ == '__main__':
    #1.创建模型
    model = MyModel().to("cuda")

    #2.定义输入数据
    x = torch.randn(10,3).to(torch.device('cuda'))

    #3 向前传播，得到输出概率
    y = model(x)

    print(y.shape)
    print(y)

    #查看模型参数
    print("第一层参数：")
    print(model.linear1.weight)
    print(model.linear1.bias)


    #2.通过遍历乍看所有参数
    print("通过遍历乍看所有参数: ")
    for name,param in model.named_parameters():
        print(name,param)

    from torchsummary import summary
    summary(model,input_size=(3,),batch_size=10,device='cuda')




