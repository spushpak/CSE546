# Q4 k-means clustering
import os
os.system('CLS')

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Net(nn.Module):
    def __init__(self, M, p, N):
        super(Net, self).__init__()

        # arguments - input channel, output channel, kernel size
        # Each input is 32 X 32 X 3 (width, height, depth or channel)
        # No. of filters M = 100 (this is same as output channel); each filter
        # is a cube (5 X 5 X 3)
        self.conv = nn.Conv2d(3, M, p)
        #self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(N, N) # both kernel size and stride is 14
        self.fc = nn.Linear(2 * 2 * M, 10)  # nn.Linear(input dim, output dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    net = Net(M=100, p=5, N=14)

    criterion = nn.CrossEntropyLoss()

    learn_rate = 0.001
    mom = 0.5
    optimizer = optim.SGD(net.parameters(), lr=learn_rate, momentum=mom)

    epoch_num = 60
    iteration = 0
    training_accuracy = []
    test_accuracy = []
    itr_seq = []

    for epoch in range(epoch_num):  # loop over the dataset multiple times
        correct = 0
        total = 0
        running_loss = 0.0
        print("\n")
        print("epoch number: ", epoch + 1)
        itr_seq.append(epoch + 1)

        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            inputs = Variable(inputs)
            labels = Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            # if i % 2000 == 1999:    # print every 2000 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 2000))
            #     running_loss = 0.0

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * (correct / total)
        print("Training accuracy: ", accuracy)
        training_accuracy.append(accuracy)
        print('Finished Training')

        # Test the network on the test data
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images = Variable(images)
                labels = Variable(labels)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * (correct / total)
        test_accuracy.append(accuracy)
        print("Test accuracy: ", accuracy)

    print("\n")
    print("Training accuracy: ", training_accuracy)
    print("Test accuracy: ", test_accuracy)

    # Plot the accuracy
    plt.clf()
    plt.plot(itr_seq, training_accuracy, label="Training accuracy" )
    plt.plot(itr_seq, test_accuracy, label="Test accuracy" )
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.title("Convolutional network: Accuracy vs Epochs")
    plt.legend(loc='best')
    #plt.show()
    plt.savefig("C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW4/Q3_convnet.png")


# Call the main() function
if __name__ == "__main__":
  main()


# Training accuracy:  [34.274, 45.153999999999996, 50.354, 53.348, 55.334, 56.647999999999996, 57.888, 58.95399999999999, 59.838, 60.548, 61.246, 61.92999999999999, 62.438, 63.108, 63.366, 63.904, 64.38000000000001, 64.522, 64.878, 65.146, 65.566, 65.75, 66.014, 66.12400000000001, 66.408, 66.57, 66.756, 67.22, 67.23, 67.396, 67.474, 67.806, 67.776, 68.094, 68.068, 68.22, 68.268, 68.476, 68.598, 68.64, 68.758, 68.78999999999999, 68.948, 69.068, 69.202, 69.42, 69.346, 69.568, 69.45599999999999, 69.572, 69.814, 69.93, 69.972, 69.92399999999999, 70.04599999999999, 70.26400000000001, 70.174, 70.26400000000001, 70.3, 70.39]
# Test accuracy:  [42.309999999999995, 47.78, 51.800000000000004, 53.68000000000001, 55.66, 55.55, 57.48, 56.61000000000001, 57.29, 59.57, 59.43000000000001, 60.550000000000004, 61.050000000000004, 61.36000000000001, 62.21, 61.760000000000005, 61.970000000000006, 62.150000000000006, 62.72, 63.74999999999999, 63.71, 62.71, 64.03999999999999, 63.73, 63.28, 64.14999999999999, 64.75, 63.580000000000005, 62.13999999999999, 64.84, 64.66, 64.08, 64.25999999999999, 64.79, 64.22, 64.25999999999999, 64.28, 65.34, 65.21000000000001, 64.75, 65.52, 65.64999999999999, 65.86999999999999, 65.24, 65.78, 63.51, 64.7, 65.78, 65.7, 65.12, 64.85, 65.52, 65.18, 65.99000000000001, 65.4, 66.22, 66.5, 66.03, 65.56, 66.53]
