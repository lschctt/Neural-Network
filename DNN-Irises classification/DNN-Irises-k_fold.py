import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from k_fold_cross_validation import get_k_fold_data


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
np.random.seed(1)
np.random.shuffle(x)
np.random.seed(1)
np.random.shuffle(y)

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


# # 构建损失函数和优化器
# model = Model()
loss = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # lr:学习率


# 设定训练过程
def train(a, train_loader, model, optimizer):
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
def test(test_loader, model):
    correct = 0
    total = 0
    with torch.no_grad():
        for t, data3 in enumerate(test_loader, 0):
            xx_test, yy_true = data3
            yy_test = model(xx_test)
            # pre为最大值的标签
            _, pre = torch.max(yy_test.data, dim=1)
            total += yy_true.size(0)
            correct += (pre == yy_true).sum().item()
            print('prediction:', pre, 'true', yy_true)
    print('Accuracy on test set: %d %%' % (100 * correct / total))
    return correct / total


# 设定预测过程
def predicted(model):
    with torch.no_grad():
        for j, data2 in enumerate(prediction_loader, 0):
            x_pre = data2
            y_pre = model.forward(x_pre)
            _, predict = torch.max(y_pre.data, dim=1)
            print('结果：', predict.tolist())


def k_fold(k, X_train, Y_train, num_epo):
    acc_list = []
    for i in range(k):

        # The model is re-instantiated each time
        model = Model()
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

        # get k_fold train data and test data (array)
        x_Train, y_Train, x_Test, y_Test = get_k_fold_data(k, i, X_train, Y_train)

        train_set = New_Dataset(x_Train, y_Train)
        train_loader = DataLoader(dataset=train_set, batch_size=1, shuffle=True, num_workers=0)
        test_set = New_Dataset(x_Test, y_Test)
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0)

        for epoch in range(num_epo):
            train(epoch, train_loader, model, optimizer)
        acc = test(test_loader, model)
        acc_list.append(acc)
        print('ith fold accuracy is:', acc)

    print(k, 'fold accuracy is:', acc_list)


if __name__ == "__main__":
    k = 10
    k_fold(k, x, y, 80)