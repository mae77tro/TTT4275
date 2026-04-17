import numpy as np
import matplotlib.pyplot as plt

N = 10
n_veq = np.arange(N)
f_actual = 0.25
fs = np.linspace(0,0.5,1000)
sigma_2 = 0.01

# Function for generating samples of x
def x_n(n):
    x = np.cos(2*np.pi*f_actual*n) + np.random.normal(0,np.sqrt(sigma_2))
    return x

cos_matrix = np.cos(2*np.pi*f[])

deltaf = 0.01
fs = np.arange(0,0.5, deltaf)
max_vals = np.zeros(5000)
maxfreqs = np.zeros(5000)

for j in range(5000):
    x = x_n(n_veq)
