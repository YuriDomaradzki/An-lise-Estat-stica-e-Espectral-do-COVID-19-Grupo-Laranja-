import mfdfa
import waipy
import statsfuncs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

def SOC(data, title, n_bins=50):
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    std = np.std(data)
    #print("mean: ", mean, " var: ", var)
    """ Computa a Taxa Local de Flutuação para cada valor da ST """ 
    Gamma = []
    
    for i in range(0,n): 
        Gamma.append((data[i] - mean)/var) 
        # Gamma.append((data[i] - mean)/std)
    
        """ Computa P[Psi_i] """
    
    
    counts, bins = np.histogram(Gamma, n_bins)
    Prob_Gamma = []
    for i in range(0, n_bins):
        Prob_Gamma.append(counts[i]/n) #plt.plot(Gamma)    
    log_Prob = np.log10(Prob_Gamma) 
    p = np.array(Prob_Gamma)
    p = p[np.nonzero(p)]
    c = counts[np.nonzero(counts)] 
    log_p = np.log10(p)
    a = (log_p[np.argmax(c)] - log_p[np.argmin(c)]) / (np.max(c) - np.min(c)) 
    b = log_Prob[0]
    y = b * np.power(10, (a*counts))
    
    
    """ Plotagem """
    
    plt.clf()
    plt.scatter(np.log10(counts), y, marker=".", color="blue")
    plt.title('SOC Country: {}'.format(title), fontsize = 16) 
    plt.xlabel('log(ni)')
    plt.ylabel('log(Yi)') 
    plt.grid()
    plt.show()
