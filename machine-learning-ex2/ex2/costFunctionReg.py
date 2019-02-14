import numpy as np
from sigmoid import *


def cost_function_reg(theta, X, y, lmd):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #

    h = sigmoid(X.dot(theta))
    cost = -(y.T.dot(np.log(h)) + (1-y).T.dot(np.log(1-h)))/m + lmd/(2*m)* (theta.T.dot(theta))

    grad0 = X.T.dot(h-y)/m
    tmp = grad0[0]
    
    grad = grad0 + lmd/(m)*theta
    grad[0] = tmp
    # ===========================================================

    return cost, grad
