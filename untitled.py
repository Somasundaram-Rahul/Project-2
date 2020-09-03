import numpy as np
import matplotlib.pyplot as plt
def f(a=2):
    fig = plt.figure()
    x = np.linspace(0,10,20)
    plt.plot(x,a*x)
    plt.show()
    
 