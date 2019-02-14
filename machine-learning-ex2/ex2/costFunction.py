import numpy as np
from sigmoid import *


def cost_function(theta, X, y):
    m = y.size

    # You need to return the following values correctly
    cost = 0
    grad = np.zeros(theta.shape)

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta
    #                You should set cost and grad correctly.
    #
    cost = -(y.T.dot(np.log(sigmoid(X.dot(theta)))) + (1-y).T.dot(np.log(1-sigmoid(X.dot(theta)))))/m

    grad = X.T.dot(sigmoid(X.dot(theta))-y)/m
    # ===========================================================

    return cost, grad
