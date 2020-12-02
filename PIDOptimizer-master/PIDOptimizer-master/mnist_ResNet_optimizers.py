import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary
from torch.autograd import Variable
from torch.optim.sgd import SGD
import pid
import os
import numpy as np
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Hyper Parameters
input_size = 784
hidden_size = 1000
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rates = [0.05, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

I = 1
I = float(I)
D = 30
D = float(D)

#logger = Logger('pid.txt', title='mnist')
#logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])

# MNIST Dataset
train_dataset = dsets.MNIST(root='./data',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data',
                           train=False,
                           transform=transforms.ToTensor())

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

BGD_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=len(train_dataset),
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Neural Network Model (1 hidden layer)
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, 3, 1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)
        self.dense = torch.nn.Linear(576, 64)
        self.maxpool = torch.nn.MaxPool2d(2, 2, 0)
        self.outlayer = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.maxpool(out)
        out = self.conv3(out)
        out = F.relu(out)
        out = out.view((out.size(0), -1))
        out = self.dense(out)
        out = self.relu(out)
        out = self.outlayer(out)

        return out


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)



def training(optimizer_sign=0, learning_rate=0.01):
    training_data = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    net = ResNet18(num_classes)
    # net = Net(input_size, hidden_size, num_classes)
    net.cuda()
    net.train()
    # Loss and Optimizer
    oldnet_sign = False
    basicgrad_sign = False
    criterion = nn.CrossEntropyLoss()
    print('optimizer_sign:' + str(optimizer_sign))
    if optimizer_sign == 0:
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    elif optimizer_sign == 1:
        optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
    elif optimizer_sign == 2:
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif optimizer_sign == 3:
        optimizer = pid.PIDOptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=D)
    elif optimizer_sign == 4:
        optimizer = pid.Adamoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    elif optimizer_sign == 5:
        optimizer = pid.RMSpropOptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001)
    elif optimizer_sign == 6:
        optimizer = pid.Momentumoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    elif optimizer_sign == 7:
        optimizer = pid.decade_PIDOptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=D)
    elif optimizer_sign == 8:
        optimizer = pid.IDoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=D)
    elif optimizer_sign == 9:
        optimizer = pid.AdapidOptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=D)
    elif optimizer_sign == 10:
        optimizer = pid.AdapidOptimizer_test(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=D)
    elif optimizer_sign == 11:
        optimizer = pid.specPIDoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=D)
        oldnet_sign = True
    elif optimizer_sign == 12:
        optimizer = pid.SVRGoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
        oldnet_sign = True
        basicgrad_sign = True
    else:
        optimizer = pid.SARAHoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
        oldnet_sign = True
        basicgrad_sign = True

    if oldnet_sign == True:
        torch.save(net, 'net.pkl')
        old_net = torch.load('net.pkl')

    # Train the Model
    for epoch in range(num_epochs):

        train_loss_log = AverageMeter()
        train_acc_log = AverageMeter()
        val_loss_log = AverageMeter()
        val_acc_log = AverageMeter()
        for i, (images, labels) in enumerate(train_loader):
            if i % 100 == 0 and basicgrad_sign == True:
                for j, (all_images, all_labels) in enumerate(BGD_loader):
                    all_images = all_images.cuda()
                    all_labels = Variable(all_labels.cuda())
                    optimizer.zero_grad()  # zero the gradient buffer
                    outputs = net(all_images)
                    train_loss = criterion(outputs, all_labels)
                    train_loss.backward()
                    params = list(net.parameters())
                    grads = []
                    for param in params:
                        grads.append(param.grad.detach())
                    optimizer.get_basicgrad(grads)
                    optimizer.step()
                    prec1, prec5 = accuracy(outputs.data, all_labels.data, topk=(1, 5))
                    train_loss_log.update(train_loss.data, all_images.size(0))
                    train_acc_log.update(prec1, all_images.size(0))
                    torch.save(net, 'net.pkl')
                    old_net = torch.load('net.pkl')
                    print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.8f'
                          % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, train_loss_log.avg,
                             train_acc_log.avg))
            # Convert torch tensor to Variable
            images = images.view(-1, 28*28).cuda()
            labels = Variable(labels.cuda())

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(images)
            train_loss = criterion(outputs, labels)
            train_loss.backward()
            if oldnet_sign == True:
                old_outputs = old_net(images)
                old_loss = criterion(old_outputs, labels)
                old_loss.backward()
                old_params = list(old_net.parameters())
                old_grads = []
                for param in old_params:
                    old_grads.append(param.grad.detach())
                optimizer.get_oldgrad(old_grads)
            optimizer.step()
            if oldnet_sign == True and optimizer_sign != 8:
                torch.save(net, 'net.pkl')
                old_net = torch.load('net.pkl')
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            train_loss_log.update(train_loss.data, images.size(0))
            train_acc_log.update(prec1, images.size(0))

            if (i + 1) % 30 == 0:
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.8f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, train_loss_log.avg,
                         train_acc_log.avg))
                training_data['train_loss'].append(train_loss_log.avg.detach().cpu().numpy())
                training_data['train_acc'].append(train_acc_log.avg.detach().cpu().numpy())

        # Test the Model
        '''
        net.eval()
        correct = 0
        loss = 0
        total = 0
        for images, labels in test_loader:
            images = images.cuda()
            labels = Variable(labels).cuda()
            outputs = net(images)
            test_loss = criterion(outputs, labels)
            val_loss_log.update(test_loss.data, images.size(0))
            prec1, prec5 = accuracy(outputs.data, labels.data, topk=(1, 5))
            val_acc_log.update(prec1, images.size(0))

        #logger.append([learning_rate, train_loss_log.avg, val_loss_log.avg, train_acc_log.avg, val_acc_log.avg])
        print('Accuracy of the network on the 10000 test images: %.8f %%' % (val_acc_log.avg))
        print('Loss of the network on the 10000 test images: %.8f' % (val_loss_log.avg))
        training_data['val_loss'].append(val_loss_log.avg.detach().cpu().numpy())
        training_data['val_acc'].append(val_acc_log.avg.detach().cpu().numpy())
        '''
    #logger.close()
    #logger.plot()
    training_data['learning_rate'] = learning_rate
    return training_data

comparing_data = []
for i in range(4):
    comparing_data.append(training(optimizer_sign=i, learning_rate=learning_rates[i]))

for data in comparing_data:
    for key, values in data.items():
        values = np.array(values)

labels = ['SGD', 'RMSprop', 'Adam', 'PID', 'Adam_self', 'RMSprop_self', 'Momentum', 'decade_PID', 'ID',
          'Adapid', 'Adapid_test', 'specPID', 'SVRG', 'SARAH']
for i in range(len(comparing_data)):
    labels[i] = labels[i] + ' learning_rate = ' + str(comparing_data[i]['learning_rate'])
for i in range(len(comparing_data)):
    plt.plot(range(len(comparing_data[i]['train_acc'])), comparing_data[i]['train_acc'], label=labels[i])
plt.legend(labels)
plt.title('DenseNet, MNIST, ' + ',i=' + str(I) + 'd=' + str(D))
plt.show()

for data in comparing_data:
    plt.plot(range(len(data['train_loss'])), data['train_loss'])

for data in comparing_data:
    plt.plot(range(len(data['val_acc'])), data['val_acc'])

for data in comparing_data:
    plt.plot(range(len(data['val_loss'])), data['val_loss'])

