import numpy as np
from sigmoid import *

def predict(theta1, theta2, x):
    # Useful values
    m = x.shape[0]
    num_labels = theta2.shape[0]
    print('shape x {}, theta1 {}, theta2 {}'.format(x.shape, theta1.shape, theta2.shape))
    # You need to return the following variable correctly
    p = np.zeros(m)

    # ===================== Your Code Here =====================
    # Instructions : Complete the following code to make predictions using
    #                your learned neural network. You should set p to a
    #                1-D array containing labels between 1 to num_labels.
    #
    x = np.c_[np.ones((m,1)), x]

    a2 = sigmoid(np.dot(x, theta1.T))

    a2 = np.c_[np.ones((m,1)), a2]

    a3 = sigmoid(np.dot(a2, theta2.T))

    p = np.argmax(a3, axis=1)+1

    return p


