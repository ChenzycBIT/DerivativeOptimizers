import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchsummary import summary
from torch.autograd import Variable
from torch.optim.sgd import SGD
import pickle
import pid
import os
import numpy as np
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from DNN_models import cifar10_CNN, cifar10_DenseNet, cifar10_ResNet18
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Hyper Parameters
input_size = 784
hidden_size = 1000
num_classes = 10
num_epochs = 5
batch_size = 100

'要进行对比实验的算法'
algorithm_labels = ['Adam_self','Momentum', 'decade_PID', 'Adapid', 'Double_Adapid', 'Restrict_Adam', 'specPID']
'每种算法所对应的学习率'


learning_rates = [0.001, 0.2, 0.2, 0.001, 0.001, 0.001, 0.2]

I = 3
I = float(I)
D = 30
D = float(D)

dataset_path = input('please input CIFAR10 path, nothing for default value \n')
if dataset_path == '':
    dataset_path = 'E:\\files\\programs\\Gitkraken\\cifar-10-batches-py\\'
for i in range(1,6):
    path = dataset_path + 'data_batch_' + str(i)
    with open(path, 'rb') as batch:
        dict = pickle.load(batch, encoding='bytes')
    if i == 1:
        images = dict[b'data']
        image_labels = dict[b'labels']
    else:
        images = np.concatenate([images, dict[b'data']], axis=0)
        image_labels = np.concatenate([image_labels, dict[b'labels']], axis=0)
path = dataset_path + 'test_batch'
with open(path, 'rb') as batch:
    dict = pickle.load(batch, encoding='bytes')
test_images = np.array(dict[b'data'])
test_image_labels = np.array(dict[b'labels'])
images = np.array(images)
image_labels = np.array(image_labels)
images = np.reshape(images, [-1, 3, 32, 32])
test_images = np.reshape(test_images, [-1, 3, 32, 32])
print(len(images))

class cifar10_dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.images = images
        self.labels = image_labels
        super(cifar10_dataset, self).__init__()
    def __getitem__(self, index):
        data = self.images[index]
        label = self.labels[index]
        return data, label
    def __len__(self):
        return len(self.images)

class cifar10_test_dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.images = test_images
        self.labels = test_image_labels
        super(cifar10_test_dataset, self).__init__()
    def __getitem__(self, index):
        data = self.images[index]
        label = self.labels[index]
        return data, label
    def __len__(self):
        return len(self.images)

train_loader = torch.utils.data.DataLoader(dataset=cifar10_dataset(), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=cifar10_test_dataset(), batch_size=batch_size, shuffle=True)
BGD_loader = torch.utils.data.DataLoader(dataset=cifar10_dataset(),batch_size=len(images),shuffle=True)


def training(model_sign=0, optimizer_sign=0, learning_rate=0.01):
    training_data = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
    if model_sign == 0:
        net = cifar10_DenseNet(num_classes)
        padding_sign = True
    elif model_sign == 1:
        net = cifar10_CNN(num_classes)
        padding_sign = False
    elif model_sign == 2:
        net = cifar10_ResNet18(num_classes)
        padding_sign = False
    else:
        raise ValueError('Not correct model sign')
    # net = Net(input_size, hidden_size, num_classes)
    net.cuda()
    net.train()
    # Loss and Optimizer
    oldnet_sign = False
    basicgrad_sign = False
    criterion = nn.CrossEntropyLoss()
    print('optimizer_sign:' + str(optimizer_sign))
    algorithm_labels = ['Adam_self', 'Momentum', 'decade_PID', 'Adapid', 'Double_Adapid', 'Restrict_Adam', 'specPID']
    if optimizer_sign == 0:
        optimizer = pid.Adamoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    elif optimizer_sign == 1:
        optimizer = pid.Momentumoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    elif optimizer_sign == 2:
        optimizer = pid.decade_PIDOptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=D/10)
    elif optimizer_sign == 3:
        optimizer = pid.AdapidOptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=0.5)
    elif optimizer_sign == 4:
        optimizer = pid.Double_Adaptive_PIDOptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I, D=0.5)
    elif optimizer_sign == 5:
        optimizer = pid.Restrict_Adamoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9)
    elif optimizer_sign == 6:
        optimizer = pid.specPIDoptimizer(net.parameters(), lr=learning_rate, weight_decay=0.0001, momentum=0.9, I=I,D=D)
        oldnet_sign = True
    else:
        raise ValueError('Not correct algorithm symbol')
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
                    all_images = all_images.cuda().float()
                    if padding_sign == True:
                        all_images = all_images.view(-1, 3072)
                    all_labels = Variable(all_labels.cuda()).long()
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
                          % (epoch + 1, num_epochs, i + 1, len(images) // batch_size, train_loss_log.avg,
                             train_acc_log.avg))
            # Convert torch tensor to Variable
            images = images.cuda().float()
            if padding_sign == True:
                images = images.view(-1, 3072)
            labels = Variable(labels.cuda()).long()

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
                      % (epoch + 1, num_epochs, i + 1, len(images) // batch_size, train_loss_log.avg,
                         train_acc_log.avg))
                training_data['train_loss'].append(train_loss_log.avg.detach().cpu().numpy())
                training_data['train_acc'].append(train_acc_log.avg.detach().cpu().numpy())

        # Test the Model
        net.eval()
        correct = 0
        loss = 0
        total = 0
        for images, labels in test_loader:
            images = images.cuda().float()
            if padding_sign == True:
                images = images.view(-1, 3072)
            labels = Variable(labels).cuda().long()
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
    #logger.close()
    #logger.plot()
    training_data['learning_rate'] = learning_rate
    return training_data

model_sign = int(input('please input model sign: \n 0 for Densenet, 1 for CNN, 2 for ResNet18 \nmodel_sign:'))


models = ['DenseNet', 'CNN', 'ResNet']


comparing_data = []

testing_algorithms = [0, 1, 2, 3, 4, 5, 6]


for i in testing_algorithms:
    comparing_data.append(training(model_sign=model_sign, optimizer_sign=i, learning_rate=learning_rates[i]))

for data in comparing_data:
    for key, values in data.items():
        values = np.array(values)

for i in range(len(testing_algorithms)):
    algorithm_labels[testing_algorithms[i]] = algorithm_labels[testing_algorithms[i]] + ' learning_rate = ' + str(comparing_data[i]['learning_rate'])
testing_labels = [algorithm_labels[i] for i in testing_algorithms]
for i in range(len(comparing_data)):
    plt.plot(range(len(comparing_data[i]['train_acc'])), comparing_data[i]['train_acc'], label=testing_labels[i])
plt.legend(testing_labels)

plt.title(models[model_sign] +  ' CIFAR10, ' + ' i=' + str(I) + 'd=' + str(D))

plt.show()

for data in comparing_data:
    plt.plot(range(len(data['train_loss'])), data['train_loss'])

for data in comparing_data:
    plt.plot(range(len(data['val_acc'])), data['val_acc'])

for data in comparing_data:
    plt.plot(range(len(data['val_loss'])), data['val_loss'])

