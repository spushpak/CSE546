import numpy as np
import matplotlib.pyplot as plt

# Load a csv of floats:
X = np.genfromtxt('C:\\local\\yelp_data\\upvote_data.csv', delimiter=",")
# Load a text file of integers:
y = np.loadtxt('C:\\local\\yelp_data\\upvote_labels.txt', dtype=np.int)
# Load a text file of strings:
featureNames = open('C:\\local\\yelp_data\\upvote_features.txt').read().splitlines()

print("X shape", X.shape)
print("y shape", y.shape)

y = np.sqrt(y)

X_train = X[0:4000, :]
X_valid = X[4000:5000, :]
X_test = X[5000:6000, :]

y = y.reshape(-1, 1)

y_train = y[0:4000, :]
y_valid = y[4000:5000, :]
y_test = y[5000:6000, :]

lam_values = 2*abs(np.dot(X_train.T, (y_train - np.mean(y_train))))
lam = np.amax(lam_values)
print("Starting lambda: ", lam)
d = X_train.shape[1]

wvec = np.full(d, 0.0)
W_est = np.full(d, 5.0)
W_mat = np.empty([0, d])
nzero_coefmat = np.empty([0, 2])
errors_train_val = np.empty([0, 4])  # hyp par, train err, val err, itr num

conv_limit = 0.5
conv_cond = np.allclose(W_est, wvec, conv_limit)
itr = 0

condition = True

while condition and itr < 10:
    hyp_par = lam/(1.5**itr)
    itr = itr + 1
    print("\nlambda: ", hyp_par)
    print("This is lambda iteration: ", itr)

    conv_cond = False
    counter = 0
    #while not conv_cond and counter <= 5:
    while not conv_cond and counter <= 5:
        counter += 1
        #print("\n This is w convergence iteration: ", counter)
        #print("what wvec is used for b: ", wvec)
        b = np.mean(y_train - np.dot(X_train, wvec))
        #print("b: ", b)
        W_est = list(wvec)

        for j in np.arange(d):
            a = 2*sum(X_train[:, j]**2)
            w_ex = np.copy(W_est)
            w_ex[j] = 0

            wx_prod = np.dot(X_train, w_ex).reshape(-1,1)
            diff = y_train - (b + wx_prod)
            c = np.asscalar(2*np.dot(X_train[:, j], diff))
            c = 2*np.dot(X_train[:, j], diff)
            #print("c: ", c)

            if c < - hyp_par:
                w = 1.*((c + hyp_par) / a)
            elif - hyp_par <= c <= hyp_par:
                w = 0.0
            else:
                w = 1.*((c - hyp_par) / a)

            wvec[j] = w

        if np.count_nonzero(wvec) == 0:
            break
        else:
            conv_cond = np.allclose(W_est, wvec, conv_limit)

    pred_train = X_train@wvec.reshape(-1, 1)
    train_err = np.mean(np.square(y_train - pred_train))

    pred_valid = X_valid@wvec.reshape(-1, 1)
    valid_err = np.mean(np.square(y_valid - pred_valid))
    errors_train_val = np.append(errors_train_val, np.array([hyp_par, train_err, valid_err, itr]).reshape(1, 4), axis=0)

    W_mat = np.append(W_mat, wvec.reshape(1, d), axis=0)
    nzero_coefmat = np.append(nzero_coefmat, np.array([hyp_par, np.count_nonzero(wvec)]).reshape(1, 2), axis=0)
    condition = np.count_nonzero(wvec) <= d


print("How many times while loop for lambda has run: ", itr)

plt.clf()
plt.plot(errors_train_val[:, 0], errors_train_val[:, 1], label="Train error")
plt.plot(errors_train_val[:, 0], errors_train_val[:, 2], label="Validation error")
plt.xlabel("Lambda")
plt.ylabel("Prediction error")
plt.title("Training and Validation error")
plt.legend(loc='best')
plt.show()
#plt.savefig('./HW2/errors_coef.png')

plt.plot(nzero_coefmat[:, 0], nzero_coefmat[:, 1], label="Non-zero coef and lambda")
plt.xlabel("Lambda")
plt.ylabel("Number of non-zero coefs")
plt.show()
#plt.savefig('./HW2/q4_nonzero_coef.png')

print("Error matrix:\n", errors_train_val)

ind = np.argmin(errors_train_val[:, 2])
print(ind)

pick_lam = errors_train_val[ind, 0]
itr_num = errors_train_val[ind, 3]

print("Best lambda: ", pick_lam)
print("Which iteration: ", int(itr_num))

print("Training error: ", errors_train_val[ind, 1])
print("Validation error: ", errors_train_val[ind, 2])

w_opt = W_mat[int(itr_num)-1, :]

pred_test = X_test@w_opt.reshape(-1, 1)
test_err = np.mean(np.square(y_test - pred_test))
print("Test error: ", test_err)

print("Optimal w:\n", w_opt)
sort_idx = np.argsort(w_opt)[::-1][0:9]
print("largest weight indices: ", sort_idx)
#w_sort = w_opt.argsort()[0:9]
print("Largest weight: ", w_opt[sort_idx])
#print("Features with largest weight: ", featureNames[sort_idx])
