import os
os.system('CLS')

import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def load_dataset():
    mndata = MNIST('C:\local\MNIST_data')
    mndata.gz = True
    x_train, labels_train = map(np.array, mndata.load_training())
    x_test, labels_test = map(np.array, mndata.load_testing())
    x_train = x_train/255.0
    x_test = x_test/255.0
    return x_train, labels_train, x_test, labels_test


def train(X, y, hyp_par):
    d = X.shape[1]
    amat = np.matmul(np.transpose(X), X) + hyp_par * np.identity(d)
    bmat = np.matmul(np.transpose(X), y)
    wmat = np.linalg.solve(amat, bmat)
    return wmat


def predict(wmat, X):
    Y_hat = np.matmul(X, wmat)
    pred_labels = Y_hat.argmax(axis=1)
    return pred_labels


# Call the load_dataset() function
X_train, train_labels, X_test, test_labels = load_dataset()

# X = X_train[0, :].reshape([28, 28])
# plt.gray()
# plt.imshow(X)
# plt.show()

# No of classes and no of features
num_class = 10  # this is k
num_features = X_train.shape[1]  # this is d = 784
train_obs = X_train.shape[0]
test_obs = X_test.shape[0]

# One hot-encoding of the train and test labels
encoded_train_labels = np.eye(num_class)[train_labels]
encoded_test_labels = np.eye(num_class)[test_labels]
print(train_labels)
print(np.eye(num_class)[5])
print("Encoded train labels: \n", encoded_train_labels)
print("Encoded test labels: \n", encoded_test_labels)

'''
# Solve for W based on training set
hyper_par = 0.0001

# Call the train function
w = train(X_train, encoded_train_labels, hyper_par)

# Call the predict function on training set
predicted_train_labels = predict(w, X_train)
print("Dimension of predicted training labels: ", predicted_train_labels.shape)
diff_labels = train_labels - predicted_train_labels   # if difference is 0, means correct prediction. Count how many 0's
train_accuracy = len(diff_labels) - np.count_nonzero(diff_labels)  # Count how many 0's
train_accuracy = train_accuracy / len(predicted_train_labels)  # divide accurate prediction by total no of obs
print("Train accuracy: ", train_accuracy)


# Call the predict function on test set
predicted_test_labels = predict(w, X_test)
diff_labels = test_labels - predicted_test_labels   # if difference is 0, means correct prediction. Count how many 0's
test_accuracy = len(diff_labels) - np.count_nonzero(diff_labels)  # Count how many 0's
test_accuracy = test_accuracy / len(predicted_test_labels)  # divide accurate prediction by total no of obs
print("Test accuracy: ", test_accuracy)


#p_list = [500, 1000, 1500, 2000, 3000, 4000, 5000, 6000]
p_list = [500, 1000, 1500, 2000]

mu = 0
sigma = np.sqrt(0.1)

tr_accuracy = []
val_accuracy = []

for p in p_list:
    gsample = np.random.normal(mu, sigma, p*num_features)
    G = np.reshape(gsample, (p, num_features))
    print("Dimension of G is: ", G.shape)

    bsample = np.random.uniform(0, 2*np.pi, p)
    bsample = bsample.reshape(1, p)
    print("Dimension of bsample is: ", bsample.shape)

    B = np.tile(bsample.transpose(), (1, train_obs))
    print("Dimension of B is: ", B.shape)

    GX = np.matmul(G, np.transpose(X_train))
    H = np.cos(GX + B)   # this is in (p X n) form
    H = np.transpose(H)  # reshaping it back to (n X p) form
    print("Dimension of H", H.shape)


    new_train_size = int(0.8*train_obs)
    val_size = int(0.6*train_obs)
    print("Train size: ", new_train_size)

    train_indx = np.random.choice(np.arange(train_obs), new_train_size, replace=False)
    train_set = H[train_indx, :]

    val_indx = np.setdiff1d(np.arange(train_obs), train_indx)
    val_set = H[val_indx, :]

    w_train = train(train_set, encoded_train_labels[train_indx], hyper_par)

    pred_train_labels = predict(w_train, train_set)
    diff_labels = train_labels[train_indx] - pred_train_labels
    new_train_accuracy = len(diff_labels) - np.count_nonzero(diff_labels)
    new_train_accuracy = new_train_accuracy / new_train_size
    tr_accuracy.append(new_train_accuracy)

    pred_val_labels = predict(w_train, val_set)
    diff1_labels = train_labels[val_indx] - pred_val_labels
    valid_accuracy = len(diff1_labels) - np.count_nonzero(diff1_labels)
    valid_accuracy = valid_accuracy / val_size
    val_accuracy.append(valid_accuracy)

    # I want this if block to be executed only once for the optimal p
    if p == 2000:             # assuming this is optimal p, need to check some sort of optimal flag
        w_opt = w_train
        p_hat = p

        # Transform the test set - needed later in part (e)
        GX_test = np.matmul(G, np.transpose(X_test))
        B_test = np.tile(bsample.transpose(), (1, test_obs))

        H_test = np.cos(GX_test + B_test)  # this is in (p X n) form
        H_test = np.transpose(H_test)  # reshaping it back to (n X p) form
        H_test_opt = H_test


# Q.5e
pred_test_labels = predict(w_opt, H_test_opt)
print("Dimension of newly predicted training labels: ", pred_test_labels.shape)
diff_test_labels = test_labels - pred_test_labels
E_test_accuracy = len(diff_test_labels) - np.count_nonzero(diff_test_labels)
E_test_accuracy = E_test_accuracy / test_obs

delta = 0.05
epsilon = np.sqrt(np.log(2 / delta) / (2 * test_obs))
lower_limit = E_test_accuracy - epsilon
upper_limit = E_test_accuracy + epsilon

print("Expectation of E_test: ", E_test_accuracy)
print("\nLower limit of E_test: ", lower_limit)
print("\nUpper limit of E_test: ", upper_limit)
'''