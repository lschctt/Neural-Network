import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# 构建数据集
# ToTensor()将shape为(H, W, C)的np.ndarray或img转为shape为(C, H, W)的tensor,并将每一个数值归一化到[0,1]
# Normalize()使用公式"(x-mean)/std"，将每个元素分布到(-1,1)
tran = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=tran)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=64)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=tran)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)


# 设计模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(784, 512)
        self.linear2 = torch.nn.Linear(512, 256)
        self.linear3 = torch.nn.Linear(256, 128)
        self.linear4 = torch.nn.Linear(128, 56)
        self.linear5 = torch.nn.Linear(56, 10)

    def forward(self, x):
        y_hat = x.view(-1, 784)
        y_hat = F.relu(self.linear1(y_hat))
        y_hat = F.relu(self.linear2(y_hat))
        y_hat = F.relu(self.linear3(y_hat))
        y_hat = F.relu(self.linear4(y_hat))
        return self.linear5(y_hat)


# 构建损失函数和优化器
model = Model()
loss = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # lr:学习率


# 设定训练过程
def train(epoch):
    num = 0
    for i, data1 in enumerate(train_loader, 0):
        (x_1, y_1) = data1

        # forward
        y_hat = model.forward(x_1)
        l = loss(y_hat, y_1)
        num += 1
        if num % 300 == 0:
            print('epoch:', epoch, 'num:', num, 'Loss:', l.item())

        # backward
        optimizer.zero_grad()
        l.backward()

        # update
        optimizer.step()


# 设定测试过程
def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data2 in test_loader:
            (x_test, labels) = data2
            y_test = model.forward(x_test)
            _, predict = torch.max(y_test.data, dim=1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    print('准确率：', (100 * correct / total), '%')


if __name__ == "__main__":
    for epoch in range(5):
        train(epoch)
    test()
