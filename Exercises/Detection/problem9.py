import scipy
import numpy as np
import matplotlib.pyplot as plt

ENR = np.linspace(0,10,100)

Pe = scipy.stats.norm.sf((1/np.sqrt(2))*np.sqrt(ENR))
    
required_ENR = 2*scipy.stats.norm.isf(10e-3)**2
print(f"Required ENR: {required_ENR}")
plt.plot(ENR,Pe)
plt.xlabel("ENR")
plt.ylabel("Pe")

plt.show()
