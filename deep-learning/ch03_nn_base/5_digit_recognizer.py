import torch
from torch import nn
from common.load_data import load_digit_data

#1.读取数据
x_train,x_test,y_train,y_test = load_digit_data()

#2.定义模型
model = nn.Sequential(
    nn.Linear(784,50),
    nn.ReLU(),
    nn.Linear(50,100),
    nn.ReLU(),
    nn.Linear(100,10),
    nn.Softmax(dim=-1),
)

#3.加在模型
state_dict = torch.load('../data/data/nn_example.pt')
model.load_state_dict(state_dict)

#4.对测试数据进行推理预测
y_pred = model(x_test)
print("预测标签形状",y_pred.shape)


#5 转换为分类标签
y_pred_label = y_pred.argmax(dim=-1)
print("预测标签形状",y_pred_label.shape)

#6评估
n = x_test.shape[0]
acc_cnt = y_pred_label.eq(y_test).sum().item()

print(f"概率{acc_cnt/n:.6f}")

