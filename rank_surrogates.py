import numpy as np

def nuclear(s):
    return s, np.ones_like(s)

def renyi(s, alpha=0.5):
    cost = 1/(1-alpha)*np.log(np.sum(s**(2*alpha)))

    fac = 2*alpha/(1-alpha)/np.sum(s**(2*alpha))
    egrad = fac*s**(2*alpha - 1)
    
    return cost, egrad, 0