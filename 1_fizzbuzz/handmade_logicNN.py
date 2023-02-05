import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logicNN(X, t):
    N = 4
    epochs = 10000
    lr = 0.1
    W1 = np.random.rand(2, N)
    W2 = np.random.rand(N, 1)

    for e in range(epochs):
        ### forward pass
        out1 = sigmoid(np.dot(X, W1))
        out2 = sigmoid(np.dot(W1, W2))

        ### backprop
        error = (t - out2)
        # chain rule
        d2 = ############## TODO
        d1 = ############## TODO
        # SGD
        W2 += lr * out1.T.dot(d2)
        W1 += lr * X.T.dot(d1)
    return out2.flatten() #np.round(out2).flatten()

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]]
)

print("OR", logicNN(X, np.array([[0,1,1,1]]).T))
# print("AND", logicNN(X, np.array([[0,0,0,1]]).T))
# print("XOR", logicNN(X, np.array([[0,1,1,0]]).T))
# print("NAND", logicNN(X, np.array([[1,1,1,0]]).T))
# print("NOR", logicNN(X, np.array([[1,0,0,0]]).T))


X = np.array([[0.1,0.1], [0.2,0.9], [0.8,0.15], [0.85,0.8]])
print("OR", logicNN(X, np.array([[0,1,1,1]]).T))

X = np.array([[0,0], [0,1], [1,0], [1,1],[0.1,0.1], [0.2,0.9], [0.8,0.15], [0.85,0.8]])
print("OR", logicNN(X, np.array([[0,1,1,1,0,1,1,1]]).T))
