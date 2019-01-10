import numpy as np
import pandas as pd
def computeCost(X, y, theta):
    m = len(X)
    inner = np.power(((X * theta.T) - y), 2)
    return float(np.sum(inner) / (2 * m))
