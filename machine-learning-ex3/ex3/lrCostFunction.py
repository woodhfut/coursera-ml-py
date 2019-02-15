import numpy as np
from sigmoid import *


def lr_cost_function(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #
    prediction = sigmoid(X.dot(theta))
    cost = -1/(m)*(y.T.dot(np.log(prediction)) + (1-y).T.dot(np.log(1-prediction))) + 1/(2*m)*theta.dot(theta.T)

    grad0 = 1/(m)*X.T.dot(prediction-y)
    tmp = grad0[0]

    grad = grad0 + lmd/(m)*theta
    grad[0] = tmp
    # =========================================================

    return cost, grad
