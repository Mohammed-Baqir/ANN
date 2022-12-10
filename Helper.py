# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 18:16:59 2021

@author: habdu
"""

def datasets_Noisy_AND_gate(N=1000, PSD = 0.01, test = True, valid = False):
    
    import numpy as np
    import numpy.random as rnd
    from matplotlib import pyplot as plt
    
    inputs = rnd.randint(0,2,(2, N))
    inputs_plus_noise = inputs + np.sqrt(PSD) * rnd.randn(2, N)
    
    w = np.ones((2,1)) # w= [1; 1]
    b = np.array([-1.5])
    Y = np.sign( np.dot(w.T, inputs) + b )
    
    if test and not valid:
        N_train = int(np.floor(N * 0.8))
        N_test = int(np.floor(N*0.2))
        X_train = inputs_plus_noise[0:2, 0:N_train]
        X_test = inputs_plus_noise[:, N_train:-1]
        Y_train = Y[:, 0:N_train]
        Y_test = Y[:, N_train:-1]
        return X_train, X_test, Y_train, Y_test
    elif test and valid:
        N_training = int(np.floor(N * 0.7))
        N_test = int(np.floor(N*0.15))
        N_validation = int(np.floor(N*0.15))
        X_train = inputs_plus_noise[:, 0:N_train]
        X_test = inputs_plus_noise[N_train:N_train+N_test]
        X_validation = inputs_plus_noise[:, N_train+N_test:-1]
        Y_train = Y[:, 0:N_train]
        Y_test = Y[N_train:N_train+N_test]
        Y_validation = Y[:, N_train+N_test:-1]
        
        return X_train, X_test, Y_train, Y_test
        
    
    
    
#    x1 = np.linspace((-1,1), 20)
#    x2 = -w[1]/w[0] * x1 - b
#    
#    
#    
#    u = np.dot(inputs_plus_noise , w)
#    v = u + b
#    y1 = np.sign( v )
#    y1 = (y1 + 1)/2
#    
#    plt.plot(inputs_plus_noise.T[0],inputs_plus_noise.T[1], '.' )
#    plt.plot(inputs.T[0],inputs.T[1], 'ro' )
#    plt.plot(x1,x2 )
#    plt.axis([-1, 2, -1, 2])
#    plt.grid()
#    
#    plt.figure()
#    plt.figure()
#    plt.stem(y1[0:100])

if "__main__" == __name__:
    X_train, X_test, Y_train, Y_test = datasets_Noisy_AND_gate(N = 1000, PSD = 0.01, test = True, valid = False)
    
