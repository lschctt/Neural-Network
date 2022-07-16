import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# prepare dataset
# ToTensor()将shape为(H, W, C)的np.ndarray或img转为shape为(C, H, W)的tensor,并将每一个数值归一化到[0,1]
# Normalize()使用公式"(x-mean)/std"，将每个元素分布到(-1,1)
tran = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='../MNIST-Classification/data/MNIST-dataset/mnist/', train=True, download=True, transform=tran)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=30)
test_dataset = datasets.MNIST(root='../MNIST-Classification/data/MNIST-dataset/mnist/', train=False, download=True, transform=tran)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=30)

# if use GPU
use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Design model
class ResBlock(torch.nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.channel = channel
        self.conv1 = torch.nn.Conv2d(channel, channel, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channel, channel, kernel_size=5, padding=2)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # Convolution layer
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3)
        # MaxPooling layer
        self.MP = torch.nn.MaxPool2d(2)
        # Residual Block
        self.RB1 = ResBlock(16)
        self.RB2 = ResBlock(32)
        # Linear layer
        self.linear1 = torch.nn.Linear(800, 200)
        self.linear2 = torch.nn.Linear(200, 10)

    def forward(self, x):
        batch = x.size(0)  # get the batch_size of x
        y = self.MP(F.relu(self.conv1(x)))
        y = self.RB1(y)
        y = self.MP(F.relu(self.conv2(y)))
        y = self.RB2(y)
        y = y.view(batch, -1)
        y = F.relu(self.linear1(y))
        y = self.linear2(y)
        return y


# construct loss and optimizer
model = Model()
loss = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.5)  # lr:学习率


# train
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


# test
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
    return correct / total


if __name__ == "__main__":
    num_epo = 15
    acc_list = []
    acc_max = 0.9917

    for epoch in range(1, num_epo):
        train(epoch)
        acc = test()
        # use to record acc
        acc_list.append(acc)

        if acc > acc_max:
            # save model at present
            torch.save(obj=model.state_dict(), f="./models/Res.pth")
            acc_max = acc

    # plot
    Epoch = list(range(1, num_epo))
    plt.plot(Epoch, acc_list, 'o-b')
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    for i in range(0, num_epo - 1):
        plt.text(Epoch[i], acc_list[i], round(acc_list[i], 4), fontsize=10, verticalalignment="bottom",
                 horizontalalignment="center")
    plt.show()



