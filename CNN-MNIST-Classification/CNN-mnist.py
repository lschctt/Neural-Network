import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# 构建数据集
# ToTensor()将shape为(H, W, C)的np.ndarray或img转为shape为(C, H, W)的tensor,并将每一个数值归一化到[0,1]
# Normalize()使用公式"(x-mean)/std"，将每个元素分布到(-1,1)
tran = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=tran)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=60)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=tran)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=60)


# 设计模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 构造卷积层，输入channel=1、输出channel=10、kernel的大小为5x5,经过该卷积层,w和h均减2
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=(5, 5))
        # 构造卷积层，输入channel=10、输出channel=20、kernel的大小为3x3,经过该卷积层,w和h均减2
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=(5, 5))

        self.conv3 = torch.nn.Conv2d(20, 30, kernel_size=(3, 3))


        # 构造池化层，大小为2x2
        self.pool = torch.nn.MaxPool2d(2)
        self.linear1 = torch.nn.Linear(120, 60)
        self.linear2 = torch.nn.Linear(60, 30)
        self.linear3 = torch.nn.Linear(30, 10)

    def forward(self, x):
        y_hat = F.relu(self.pool(self.conv1(x)))
        y_hat = F.relu(self.pool(self.conv2(y_hat)))
        y_hat = F.relu(self.conv3(y_hat))
        y_hat = y_hat.view(-1, 120)
        y_hat = F.relu(self.linear1(y_hat))
        y_hat = F.relu(self.linear2(y_hat))
        return self.linear3(y_hat)


# 构建损失函数和优化器
model = Model()
loss = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.5)  # lr:学习率


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
    # a.append(correct/total)
    print('准确率：', (100 * correct / total), '%')


if __name__ == "__main__":
    # accuracy = []
    # Epoch = []
    # Epoch.append(i)
    for epoch in range(10):
        train(epoch)
    test()

    '''
    plt.plot(Epoch, accuracy)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.show()
    '''