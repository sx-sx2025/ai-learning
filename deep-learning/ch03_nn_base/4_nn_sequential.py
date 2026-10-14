import device
import torch
from torch import nn









#测试
if __name__ == '__main__':
    #1.创建模型
    model = nn.Sequential(
        nn.Linear(3,4),
        nn.Tanh(),
        nn.Linear(4,4),
        nn.ReLU(),
        nn.Linear(4,2),
        nn.Softmax(dim=-1),

    ).to(device = torch.device('cuda'))

    #2.定义输入数据
    x = torch.randn(10,3).to(torch.device('cuda'))

    #3 向前传播，得到输出概率
    y = model(x)

    print(y.shape)
    print(y)
    #
    # #查看模型参数
    # # print("第一层参数：")
    # # print(model.linear1.weight)
    # # print(model.linear1.bias)
    #
    #
    # #2.通过遍历乍看所有参数
    # print("通过遍历乍看所有参数: ")
    # for name,param in model.named_parameters():
    #     print(name,param)
    #
    # from torchsummary import summary
    # summary(model,input_size=(3,),batch_size=10,device='cuda')
    #
    # #3. 状态字典
    # print("---------------状态字典---------------")
    #
    # from torchsummary import summary
    # summary(model,input_size=(3,),batch_size=10,device='cuda')

    #保存
    #torch.save(model.state_dict(),'model.pth')

    # 加载
    state_dict = torch.load('./model.pth')
    model.load_state_dict(state_dict)
    # 构造输入：2 个样本，每个 3 个特征
    x_new = torch.tensor([[1.2,3.4,3.2],[2.5,-1.3,0.8]]).to(torch.device('cuda'))

    y_pred = model(x_new)
    print("预测分类的概率：", y_pred)

    y_label = y_pred.argmax(dim=-1)
    print("预测分类的标签：", y_label)