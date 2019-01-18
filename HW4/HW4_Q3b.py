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
    def __init__(self, M):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, M)  # nn.Linear(input dim, hidden dim)
        #self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(M, 10)  # nn.Linear(hidden dim, output dim)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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


    net = Net(M=500)

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
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy")
    plt.title("One hidden layer: Accuracy vs Epochs")
    plt.legend(loc='best')
    #plt.show()
    plt.savefig("C:/GoogleDrivePushpakUW/UW/6thYear/CSE546/HW4/Q3_onehidden.png")


# Call the main() function
if __name__ == "__main__":
  main()

# For M=150
# Training accuracy:  [39.106, 46.538000000000004, 49.919999999999995, 52.138, 54.112, 55.669999999999995, 56.934, 58.199999999999996, 59.282000000000004, 60.422, 61.507999999999996, 62.29599999999999, 63.14999999999999, 64.302, 65.094, 65.69800000000001, 66.74600000000001, 67.394, 67.948, 68.65599999999999, 69.374, 69.91000000000001, 70.56, 71.092, 71.834, 72.318, 72.99, 73.588, 74.5, 74.41, 75.298, 75.57000000000001, 76.42, 76.708, 76.974, 77.714, 78.024, 78.49199999999999, 78.888, 79.594, 79.908, 80.338, 80.542, 80.874, 81.23, 81.57600000000001, 82.1, 82.706, 82.914, 83.184, 83.854, 83.816, 84.524, 84.834, 85.104, 85.20400000000001, 85.49, 85.918, 86.196, 86.388]
# Test accuracy:  [44.86, 47.29, 49.76, 51.05, 51.11, 51.11, 51.5, 52.26, 52.559999999999995, 52.17, 52.42, 52.59, 52.849999999999994, 52.449999999999996, 52.480000000000004, 51.480000000000004, 52.65, 51.739999999999995, 52.290000000000006, 51.59, 52.32, 52.01, 51.77, 52.459999999999994, 51.89, 51.42, 51.6, 51.53, 51.12, 51.27, 51.55, 51.32, 51.370000000000005, 50.28, 51.21, 50.62, 50.9, 51.4, 50.839999999999996, 50.68, 50.51, 50.18, 51.04, 50.260000000000005, 50.33, 50.23, 49.84, 50.73, 50.2, 50.370000000000005, 50.21, 49.370000000000005, 50.38, 49.61, 50.13999999999999, 50.59, 49.91, 49.76, 49.7, 49.59]
# [Finished in 4370.336s]


# For M=500
# Training accuracy:  [40.012, 47.3, 51.205999999999996, 53.878, 56.03, 58.138, 59.952000000000005, 61.565999999999995, 62.944, 64.646, 65.812, 67.096, 68.582, 69.894, 71.054, 72.098, 73.42, 74.74199999999999, 76.01599999999999, 76.97, 77.964, 79.392, 80.296, 81.608, 82.244, 83.314, 84.226, 85.10799999999999, 85.712, 86.764, 87.586, 88.634, 89.13799999999999, 89.79599999999999, 90.424, 90.926, 91.694, 92.432, 92.828, 93.518, 93.974, 94.57799999999999, 95.052, 95.192, 95.72800000000001, 96.03399999999999, 96.50800000000001, 96.836, 96.952, 97.35000000000001, 97.7, 98.07000000000001, 98.156, 98.422, 98.424, 98.494, 98.7, 98.904, 99.154, 99.188]
# Test accuracy:  [45.62, 48.44, 50.72, 50.91, 52.6, 52.580000000000005, 52.68000000000001, 53.400000000000006, 53.44, 53.68000000000001, 53.11, 52.239999999999995, 53.339999999999996, 53.74, 53.81, 53.76, 54.26, 53.339999999999996, 53.55, 53.32, 52.769999999999996, 52.64, 54.400000000000006, 53.300000000000004, 53.7, 52.980000000000004, 53.080000000000005, 52.65, 53.26, 52.739999999999995, 52.5, 52.669999999999995, 53.0, 53.010000000000005, 53.66, 51.949999999999996, 51.690000000000005, 52.459999999999994, 51.43, 52.410000000000004, 52.32, 52.66, 52.59, 53.190000000000005, 52.39, 53.33, 52.080000000000005, 53.0, 52.370000000000005, 52.65, 52.93, 52.94, 52.65, 52.949999999999996, 53.12, 52.17, 53.03, 53.05, 53.44, 52.64]
