import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import torch.optim as optim


class ConvExtractor(nn.Module):
    def __init__(self, ninput, noutput, kernel_size, stride=1):
        super(ConvExtractor, self).__init__()
        self.padding = int((kernel_size - 1) / 2)
        self.layer1 = nn.Sequential(
            nn.Conv2d(ninput, noutput, kernel_size, stride=stride, padding=self.padding),
            nn.BatchNorm2d(noutput),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(noutput, 32, kernel_size, stride=stride, padding=self.padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.out_dim = 32 * 14 * 5

    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = out.view(len(input), -1)
        return out


class CnnSVM(nn.Module):
    def __init__(self, ninput, noutput, kernel_size, stride=1, num_class=12):
        self.extractor = ConvExtractor(ninput, noutput, kernel_size, stride)
        self.fc = nn.Linear(self.extractor.out_dim, num_class)

    def forward(self, input):
        feature = self.extractor(input)
        out = self.fc(feature)
        return out


def train(net, lr, m, num_epoch, trainloader, device, testloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=m)
    losses = []
    count = 0
    test_acc = []
    train_acc = []

    for epoch in tqdm(range(num_epoch)):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            per_batch = 500
            count += 1
            if count % per_batch == 0:    # print every 2000 mini-batches
                losses.append(running_loss / per_batch)
                tmp_test = test(net, testloader, device)
                tmp_train = test(net, trainloader, device)
                print('[%d, %5d] loss: %.3f\t train_acc: %.3f \t test_acc: %.3f' %
                      (epoch + 1, i + 1, running_loss / per_batch, tmp_train, tmp_test))
                running_loss = 0.0

        train_acc.append(test(net, trainloader, device))
        test_acc.append(test(net, testloader, device))
    print("train_acc : " + str(train_acc))
    print("test_acc : " + str(test_acc))
    # plt.plot(losses)
    # plt.xlabel("itr.")
    # plt.ylabel("loss")
    # plt.show()
    print('Finished Training')
    return net


def test(net, testloader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print('Accuracy of the network on the 10000 test images: %d %%' % (
    #     100 * correct / total))
    rate = 100 * correct / total
