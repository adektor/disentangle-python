import numpy as np

def nuclear(s):
    return s, np.ones_like(s)

def renyi(s, alpha=2):
    return s**alpha, alpha*(s**(alpha-1)), alpha*(alpha-1)*(s**(alpha-2))