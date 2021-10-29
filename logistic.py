import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

N = 1000
D = 10

#sample data
#create a random dataset
rng = np.random.RandomState(42)
#generate a matrix data containing N samples and D features
data, y = rng.randn(N, D), rng.randint(0, 2, size=N)
#split data in train & test set
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data, y, random_state=10, test_size=0.3)

# randomly initialize weights
W = np.random.randn(D)
b = 0

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)

def accuracy_score(y, y_pred):
    return np.mean(y == y_pred)

def cross_entropy(y, y_prob):
    return -np.mean(y*np.log(y_prob) + (1 - y)*np.log(1 - y_prob))

train_costs = []
test_costs = []
learning_rate = 0.001

for i in range(100):
    Ytrain_prob = forward(Xtrain, W, b)
    Ytest_prob = forward(Xtest, W, b)

    train_error = cross_entropy(Ytrain, Ytrain_prob)
    test_error = cross_entropy(Ytest, Ytest_prob)
    train_costs.append(train_error)
    test_costs.append(test_error)

    # gradient descent
    W -= learning_rate*Xtrain.T.dot(Ytrain_prob - Ytrain)
    b -= learning_rate*(Ytrain_prob - Ytrain).sum()

    if i % 10 == 0:
        print(i, train_error, test_error)

print("train accuracy: {}".format(accuracy_score(Ytrain, np.round(Ytrain_prob))))
print("test accuracy: {}".format(accuracy_score(Ytest, np.round(Ytest_prob))))

#plot error
legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()