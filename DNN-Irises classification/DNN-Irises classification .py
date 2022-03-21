import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


# 构建训练数据集
class New_Dataset(Dataset):
    def __init__(self, xx, yy):
        super().__init__()
        # 选取指标列
        x1 = xx
        x2 = x1.astype(np.float32)
        y1 = yy
        y2 = y1.astype(np.float32)

        self.len = len(xx)
        self.x_data = torch.from_numpy(x2)
        self.y_data = torch.from_numpy(y2)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 构建预测数据集
class New_Dataset_test(Dataset):
    def __init__(self, filepath, sheet):
        super().__init__()
        # 选取指标列
        data = pd.read_excel(filepath, sheet_name=sheet)
        x1 = data.values[:, [0, 1, 2, 3]]
        x2 = x1.astype(np.float32)

        self.len = data.values.shape[0]
        self.x_data = torch.from_numpy(x2)

    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len


data_set = pd.read_excel('./data/Irises classification.xlsx', sheet_name='train')
x = data_set.values[:, [0, 1, 2, 3]]
y = data_set.values[:, [5]]
# 划分训练集和测试集
# x:待划分样本数据  y:待划分样本数据的标签  test_size:划分测试集的比例  random_state:随机种子，赋值为0时每次都不一样
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

train_set = New_Dataset(x_train, y_train)
train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=0)
test_set = New_Dataset(x_test, y_test)
test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0)

prediction_set = (New_Dataset_test('./data/Irises classification.xlsx', 'prediction'))
prediction_loader = DataLoader(dataset=prediction_set, batch_size=1, shuffle=False, num_workers=0)


# 设计模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 3)

    def forward(self, u):
        u = F.relu(self.linear1(u))
        return self.linear2(u)


# 构建损失函数和优化器
model = Model()
loss = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # lr:学习率


# 设定训练过程
def train(a):
    num = 0
    for i, data1 in enumerate(train_loader, 0):
        (x_1, y_1) = data1
        # print(x_1)

        # forward
        y_hat = model.forward(x_1)
        y_1 = y_1.squeeze(0)
        l = loss(y_hat, y_1.long())
        num += 1
        if num % 2 == 0:
            print('epoch:', a, 'num:', num, 'Loss:', l.item())

        # backward
        optimizer.zero_grad()
        l.backward()

        # update
        optimizer.step()


# 设定训练过程
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for t, data3 in enumerate(test_loader,0):
            xx_test, yy_true = data3
            yy_test = model(xx_test)
            # pre为最大值的标签
            _, pre = torch.max(yy_test.data, dim=1)
            total += yy_true.size(0)
            correct += (pre == yy_true).sum().item()
            print('prediction:', pre, 'true', yy_true)
    print('Accuracy on test set: %d %%' % (100 * correct / total))


# 设定预测过程
def predicted():
    with torch.no_grad():
        for j, data2 in enumerate(prediction_loader, 0):
            x_pre = data2
            y_pre = model.forward(x_pre)
            _, predict = torch.max(y_pre.data, dim=1)
            print('结果：', predict.tolist())


if __name__ == "__main__":
    for epoch in range(100):
        train(epoch)
    test()
    predicted()
