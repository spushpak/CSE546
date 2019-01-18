# Q1
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

os.system('CLS')

def cordinate(eigen_val, eigen_vec, origin):
    x = np.sqrt(eigen_val*(eigen_vec[0]**2/(eigen_vec[0]**2 + eigen_vec[1]**2)))
    y = x*(eigen_vec[1]/eigen_vec[0])

    x_final = origin[0] + x
    y_final = origin[1] + y
    return x_final, y_final

# Main program
mu1 = np.array([1, 2]).reshape(2, 1)
sigma1 = np.array([[1, 0], [0, 2]])

mu2 = np.array([-1, 1]).reshape(2, 1)
sigma2 = np.array([[2, -1.8], [-1.8, 2]])

mu3 = np.array([2, -2]).reshape(2, 1)
sigma3 = np.array([[3, 1], [1, 2]])

# Q1a
# Compute eigen value and eigen vectors to find the square root of the sigma
# matrices: [Sigma^(1/2) = P D^(1/2) P^(-1)]
# v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
eigenval1, eigenvec1 = np.linalg.eig(sigma1)
eigenval2, eigenvec2 = np.linalg.eig(sigma2)
eigenval3, eigenvec3 = np.linalg.eig(sigma3)

sigma1_sqrt = eigenvec1.dot(np.sqrt(np.diag(eigenval1))). \
dot(np.linalg.inv(eigenvec1))
sigma2_sqrt = eigenvec2.dot(np.sqrt(np.diag(eigenval2))). \
dot(np.linalg.inv(eigenvec2))
sigma3_sqrt = eigenvec3.dot(np.sqrt(np.diag(eigenval3))). \
dot(np.linalg.inv(eigenvec3))

n = 100

Z1 = np.random.randn(2, n)
X1 = mu1 + np.matmul(sigma1_sqrt, Z1)

Z2 = np.random.randn(2, n)
X2 = mu2 + np.matmul(sigma2_sqrt, Z2)

Z3 = np.random.randn(2, n)
X3 = mu3 + np.matmul(sigma3_sqrt, Z3)

# Plot the scatter
dirname = os.path.dirname(__file__)
# print(dirname)
filename = os.path.join(dirname, 'plots/X1_scatter.png')

plt.clf()
fig, ax = plt.subplots(2, 2)
fig.subplots_adjust(hspace=.5)
ax[0, 0].scatter(X1[0, :], X1[1, :], marker='v') #row=0, col=0
ax[0, 0].set_title("X1 Scatter")
ax[0, 1].scatter(X2[0, :], X2[1, :], marker='v') #row=0, col=1
ax[0, 1].set_title("X2 Scatter")
ax[1, 0].scatter(X3[0, :], X3[1, :], marker='v') #row=1, col=0
ax[1, 0].set_title("X3 Scatter")
plt.savefig(filename)

mu1_hat = np.mean(X1, axis=1)
mu2_hat = np.mean(X2, axis=1)
mu3_hat = np.mean(X3, axis=1)

sigma1_hat = np.cov(X1)
sigma2_hat = np.cov(X2)
sigma3_hat = np.cov(X3)

# print("sigma1_hat: ", sigma1_hat)
# print("sigma2_hat: ", sigma2_hat)
# print("sigma3_hat: ", sigma3_hat)

eigval1, eigvec1 = np.linalg.eig(sigma1_hat)
eigval2, eigvec2 = np.linalg.eig(sigma2_hat)
eigval3, eigvec3 = np.linalg.eig(sigma3_hat)

c1 = max(np.amax(abs(X1)), np.amax(abs(X2)), np.amax(abs(X3)))
c1 = int(c1) + 1

lambda11 = -np.sort(-eigval1)[0]
lambda12 = -np.sort(-eigval1)[1]
temp_diff1 = X1 - mu1_hat.reshape(2, 1)

X1_hat = np.array([[eigvec1[:, 0].reshape(1, 2).dot(temp_diff1)/ \
np.sqrt(lambda11)], [eigvec1[:, 1].reshape(1, 2).dot(temp_diff1)/ \
np.sqrt(lambda12)]])

X1_hat = np.reshape(X1_hat, (-1, n))

lambda21 = -np.sort(-eigval2)[0]
lambda22 = -np.sort(-eigval2)[1]
temp_diff2 = X2 - mu2_hat.reshape(2, 1)

X2_hat = np.array([[eigvec2[:, 0].reshape(1, 2).dot(temp_diff2)/ \
np.sqrt(lambda21)], [eigvec2[:, 1].reshape(1, 2).dot(temp_diff2)/ \
np.sqrt(lambda22)]])

X2_hat = np.reshape(X2_hat, (-1, n))

lambda31 = -np.sort(-eigval3)[0]
lambda32 = -np.sort(-eigval3)[1]
temp_diff3 = X3 - mu3_hat.reshape(2, 1)

X3_hat = np.array([[eigvec3[:, 0].reshape(1, 2).dot(temp_diff3)/ \
np.sqrt(lambda31)], [eigvec3[:, 1].reshape(1, 2).dot(temp_diff3)/ \
np.sqrt(lambda32)]])

X3_hat = np.reshape(X3_hat, (-1, n))

c2 = max(np.amax(abs(X1_hat)), np.amax(abs(X2_hat)), np.amax(abs(X3_hat)))
c2 = int(c2) + 1
c = max(c1, c2)

# X1 - Computing new coordinates using the given condition in question
vec = eigvec1[:, 0]
x_cord1, y_cord1 = cordinate(lambda11, vec, mu1_hat)

vec = eigvec1[:, 1]
x_cord2, y_cord2 = cordinate(lambda12, eigvec1[:, 1], mu1_hat)

plt.clf()
filename = os.path.join(dirname, 'plots/X1.png')
plt.plot(X1[0, :], X1[1, :], 'v')
plt.plot([mu1_hat[0], x_cord1], [mu1_hat[1], y_cord1], color='r', linewidth=2)
plt.plot([mu1_hat[0], x_cord2], [mu1_hat[1], y_cord2], color='g', linewidth=2)
plt.xlim(-c, c)
plt.ylim(-c, c)
plt.title('X1')
plt.savefig(filename)

plt.clf()
filename = os.path.join(dirname, 'plots/X1_hat.png')
plt.plot(X1[0, :], X1[1, :], 'v', label='X1')
plt.plot(X1_hat[0, :], X1_hat[1, :], 'o', label='X1_hat')
plt.plot([mu1_hat[0], x_cord1], [mu1_hat[1], y_cord1], color='r', linewidth=2)
plt.plot([mu1_hat[0], x_cord2], [mu1_hat[1], y_cord2], color='g', linewidth=2)
plt.xlim(-c, c)
plt.ylim(-c, c)
plt.legend(loc='best')
plt.savefig(filename)

# X2 - Computing new coordinates using the given condition in question
vec = eigvec2[:, 0]
x_cord1, y_cord1 = cordinate(lambda21, vec, mu2_hat)

vec = eigvec2[:, 1]
x_cord2, y_cord2 = cordinate(lambda22, eigvec2[:, 1], mu2_hat)

plt.clf()
filename = os.path.join(dirname, 'plots/X2.png')
plt.plot(X2[0, :], X2[1, :], 'v')
plt.plot([mu2_hat[0], x_cord1], [mu2_hat[1], y_cord1], color='r', linewidth=2)
plt.plot([mu2_hat[0], x_cord2], [mu2_hat[1], y_cord2], color='g', linewidth=2)
plt.xlim(-c, c)
plt.ylim(-c, c)
plt.title('X2')
plt.savefig(filename)

plt.clf()
filename = os.path.join(dirname, 'plots/X2_hat.png')
plt.plot(X2[0, :], X2[1, :], 'v', label='X2')
plt.plot(X2_hat[0, :], X2_hat[1, :], 'o', label='X2_hat')
plt.plot([mu2_hat[0], x_cord1], [mu2_hat[1], y_cord1], color='r', linewidth=2)
plt.plot([mu2_hat[0], x_cord2], [mu2_hat[1], y_cord2], color='g', linewidth=2)
plt.xlim(-c, c)
plt.ylim(-c, c)
plt.legend(loc='best')
plt.savefig(filename)


# X3 - Computing new coordinates using the given condition in question
vec = eigvec3[:, 0]
x_cord1, y_cord1 = cordinate(lambda31, vec, mu3_hat)

vec = eigvec3[:, 1]
x_cord2, y_cord2 = cordinate(lambda32, vec, mu3_hat)

plt.clf()
filename = os.path.join(dirname, 'plots/X3.png')
plt.plot(X3[0, :], X3[1, :], 'v', label='X3')
plt.plot([mu3_hat[0], x_cord1], [mu3_hat[1], y_cord1], color='r', linewidth=2)
plt.plot([mu3_hat[0], x_cord2], [mu3_hat[1], y_cord2], color='g', linewidth=2)
plt.xlim(-c, c)
plt.ylim(-c, c)
plt.title('X3')
plt.savefig(filename)

plt.clf()
filename = os.path.join(dirname, 'plots/X3_hat.png')
plt.plot(X3[0, :], X3[1, :], 'v', label='X3')
plt.plot(X3_hat[0, :], X3_hat[1, :], 'o', label='X3_hat')
plt.plot([mu3_hat[0], x_cord1], [mu3_hat[1], y_cord1], color='r', linewidth=2)
plt.plot([mu3_hat[0], x_cord2], [mu3_hat[1], y_cord2], color='g', linewidth=2)
plt.xlim(-c, c)
plt.ylim(-c, c)
plt.legend(loc='best')
plt.savefig(filename)

print("This is the end of code")
