import numpy as np

def normal_eqn(X, y):
    theta = np.zeros((X.shape[1], 1))

    # ===================== Your Code Here =====================
    # Instructions : Complete the code to compute the closed form solution
    #                to linear regression and put the result in theta
    #
    xt =X.T
    theta = np.linalg.pinv(xt.dot(X)).dot(xt).dot(y)
    return theta
