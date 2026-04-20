import numpy as np
import matplotlib.pyplot as plt

N = 10**4
mu = 0
sigma = 1
lbda = 0.3

mus = np.ones(N)*mu
sigmas = np.ones(N)*sigma

x = np.random.normal(mus,sigmas)
y = np.linspace(0,N,N)
plt.scatter(x[:100],y[:100],marker="x")
lbdas = np.ones(N)*lbda
lin = np.linspace(-1,101,N)
plt.plot(lbdas,lin,color = "black")
plt.show()



