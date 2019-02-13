import numpy as np


def compute_cost(X, y, theta):
    # Initialize some useful values
    m = y.size
    cost = 0

    # ===================== Your Code Here =====================
    # Instructions : Compute the cost of a particular choice of theta.
    #                You should set the variable "cost" to the correct value.
    

    # ==========================================================

    # cost of linear regression 
    # J = (X*theta-y)'*(x*theta-y)/(2*m)
    # or J = (x*theta-y)**2/(2*m)

    #cost = sum((X.dot(theta)-y)**2)/(2*m)
    #cost = (X.dot(theta)-y).transpose().dot(X.dot(theta)-y)/(2*m)
    diff = (np.dot(X, theta)-y)
    cost = np.dot(diff, diff.T)/(2*m)
    return cost
