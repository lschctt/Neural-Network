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

train_dataset = datasets.MNIST(root='../MNIST-Classification/data/MNIST-dataset/mnist/', train=True, download=True, transform=tran)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=60)
test_dataset = datasets.MNIST(root='../MNIST-Classification/data/MNIST-dataset/mnist/', train=False, download=True, transform=tran)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=60)

# if use GPU
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义Inception类
class Inception(torch.nn.Module):
    def __init__(self, inchannel):
        super(Inception, self).__init__()
        # branch1 'Pool'
        self.b1_pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.b1_conv = torch.nn.Conv2d(inchannel, 24, kernel_size=(1, 1))

        # branch2 'conv_1x1'
        self.b2_conv1x1 = torch.nn.Conv2d(inchannel, 16, kernel_size=(1, 1))

        # branch3 'conv_5x5'
        self.b3_conv1x1 = self.b2_conv1x1
        self.b3_conv5x5 = torch.nn.Conv2d(16, 24, kernel_size=(5, 5), padding=2)

        # branch4 'conv_3x3'
        self.b4_conv1x1 = self.b2_conv1x1
        self.b4_conv3x3_1 = torch.nn.Conv2d(16, 20, kernel_size=(3, 3), padding=1)
        self.b4_conv3x3_2 = torch.nn.Conv2d(20, 24, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        y1 = self.b1_conv((self.b1_pool(x)))
        y2 = self.b2_conv1x1(x)
        y3 = self.b3_conv1x1(x)
        y3 = self.b3_conv5x5(y3)
        y4 = self.b4_conv1x1(x)
        y4 = self.b4_conv3x3_1(y4)
        y4 = self.b4_conv3x3_2(y4)
        output = [y1, y2, y3, y4]
        # torch.cat用来拼接tensor,此时数据为(b,c,w,h).维度从0开始,所以channel是维度1
        return torch.cat(output, dim=1)


# 设计模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(88, 20, kernel_size=(5, 5))

        self.incep1 = Inception(10)
        self.incep2 = Inception(20)

        self.pool = torch.nn.MaxPool2d(2)

        self.linear1 = torch.nn.Linear(1408, 600)
        self.linear2 = torch.nn.Linear(600, 10)

    def forward(self, x):
        y_hat = F.relu(self.pool(self.conv1(x)))
        y_hat = F.relu(self.incep1(y_hat))
        y_hat = F.relu(self.pool(self.conv2(y_hat)))
        y_hat = F.relu(self.incep2(y_hat))
        y_hat = y_hat.view(-1, 1408)
        y_hat = F.relu(self.linear1(y_hat))
        y_hat = self.linear2(y_hat)
        return y_hat


model = Model()
model = model.to(device)
loss = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 设定训练过程
def train( epoch ):
    num = 0
    for i, data1 in enumerate(train_loader, 0):
        (x_1, y_1) = data1
        (x_1, y_1) = (x_1.to(device), y_1.to(device))

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
            (x_test, labels) = (x_test.to(device), labels.to(device))
            y_test = model.forward(x_test)
            _, predict = torch.max(y_test.data, dim=1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    # a.append(correct/total)
    print('准确率：', (100 * correct / total), '%')
    return correct / total


if __name__ == "__main__":
    num_epo = 30
    acc_list = []
    acc_max = 0.9906
    for epoch in range(1, num_epo):
        train(epoch)
        acc = test()
        acc_list.append(acc)
        if acc > acc_max:
            torch.save(obj=model.state_dict(), f="./models/Google_Net.pth")
            acc_max = acc

    Epoch = list(range(1, num_epo))
    plt.plot(Epoch, acc_list, 'o-b')
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    for i in range(0, num_epo-1):
        plt.text(Epoch[i], acc_list[i], round(acc_list[i], 3), fontsize=10, verticalalignment="bottom",
                 horizontalalignment="center")
    plt.show()