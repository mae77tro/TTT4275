import numpy as np
import matplotlib.pyplot as plt

A = 1
sigma_2_values = np.logspace(0,-3,20)
N = 1000
M = 1000

median_variances = []
crlb_results = []

for var in sigma_2_values:
    sigma = np.sqrt(var)
    noise = np.random.normal(0,sigma,size=(N,M))
    a = A + noise
    medians = np.median(a,axis=0)
    median_variances.append(np.var(medians))

    crlb = var/N
    crlb_results.append(crlb)




plt.loglog(sigma_2_values,median_variances, label='Var(Median Estimator)')
plt.loglog(sigma_2_values,crlb_results, label="CRLB (Mean Estimator)")

plt.xlabel("Noise Variance$\sigma^2$")
plt.ylabel("Variance of estimator")
plt.legend()
plt.show()