import numpy as np
import matplotlib.pyplot as plt

A = 1
sigma_2_values = np.logspace(0,-3,20)
N = 1000
M = 1000
epsilon = 1e-2

median_variances = []
crlb_results = []
means_variances = []

for var in sigma_2_values:
    sigma = np.sqrt(var)
    regular_noise = np.random.normal(0,sigma,size=(N,M))
    shot_noise = np.abs(np.random.normal(0,np.sqrt(20),size=(N,M)))

    is_shot_noise = np.random.rand(N,M) < epsilon

    noise = np.where(is_shot_noise,shot_noise,regular_noise)
    a = A + noise
    medians = np.median(a,axis=0)
    median_variances.append(np.var(medians))

    means = np.average(a,axis=0)
    means_variances.append(np.var(means))


    crlb = var/N
    crlb_results.append(crlb)




plt.loglog(sigma_2_values,median_variances, label='Var(Median Estimator)')
plt.loglog(sigma_2_values,means_variances, label="Var(Mean Estimator)")
plt.loglog(sigma_2_values,crlb_results,"r--", label="CRLB of Mean Estimator")
plt.xlabel("Noise Variance $\sigma^2$")
plt.ylabel("Variance of estimator")
plt.legend()
plt.show()