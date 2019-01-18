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
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        #x = x.view(-1, 3 * 32 * 32)
        x = self.fc1(x)
        return x

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


    net = Net()

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

            inputs = Variable(inputs.view(-1, 3 * 32 * 32))
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
                images = Variable(images.view(-1, 3 * 32 * 32))
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
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.title("Zero hidden layer / Logistic Regression: Accuracy vs Iteration")
    plt.legend(loc='best')
    #plt.show()
    plt.savefig("C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW4/Q3_nohidden.png")


# Call the main() function
if __name__ == "__main__":
  main()

# Training accuracy:  [36.516, 39.672000000000004, 40.678, 41.008, 41.382000000000005, 41.666, 41.874, 41.74, 42.193999999999996, 42.3, 42.426, 42.556, 42.768, 42.622, 42.912, 42.802, 43.074, 43.102000000000004, 43.047999999999995, 43.246, 43.064, 43.318, 43.222, 43.626, 43.612, 43.366, 43.416, 43.512, 43.669999999999995, 43.624, 43.541999999999994, 43.768, 43.672, 43.94, 43.972, 43.964, 43.778, 44.006, 43.87, 43.938, 44.024, 44.275999999999996, 44.112, 44.15, 44.226, 44.07, 44.272, 44.192, 44.42, 44.316, 44.352000000000004, 44.274, 44.314, 44.391999999999996, 44.372, 44.376, 44.452000000000005, 44.454, 44.556000000000004, 44.484]
# Test accuracy:  [39.269999999999996, 38.12, 39.519999999999996, 39.08, 40.239999999999995, 39.129999999999995, 39.7, 39.03, 38.07, 38.34, 38.81, 40.02, 38.04, 38.47, 38.87, 38.43, 38.37, 37.93, 38.78, 38.1, 39.14, 38.83, 38.67, 38.59, 39.35, 39.64, 38.53, 38.99, 39.61, 38.34, 38.93, 37.72, 37.01, 39.34, 37.54, 38.769999999999996, 38.24, 37.78, 39.160000000000004, 37.05, 37.32, 37.5, 38.56, 37.76, 38.53, 38.15, 39.1, 39.37, 38.65, 37.55, 38.45, 39.0, 38.31, 38.34, 37.84, 39.019999999999996, 38.440000000000005, 37.56, 38.76, 38.21]
