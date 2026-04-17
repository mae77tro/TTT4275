import numpy as np
import matplotlib.pyplot as plt
N_values = [10,50,100,150]
f = np.linspace(0,0.5,1000)

results = {}

# We want to find the MLE for the observations x[n] = cos(2*pi*f*n) + w
# Where w ~ N(0,1)  
# For this we have to maximize the log likelyhood function with respect to f
# The log likelyhood function is N/2*ln(2pi) - 1/2 * the sum from 0 to N of (x[n]-cos(2pifn))**2 which simplifies to
# For maximizing this we need to maximize the sum from 0 to N of (x[n]-cos(2pifn))**2 with respect to f
# Which can be written as: 
# f_hat = argmax the sum from 0 to N of x^2[n] - 2*x[n]-cos(2pifn) + cos^2(2pifn)
for N in N_values:
    n = np.arange(N)

    results[N] = np.array([np.sum(np.cos(2*np.pi*fk*n)**2) for fk in f])

plt.figure(1)
for N_value in N_values:
    plt.plot(f,results[N_value],label = N_value)
plt.legend()


#This plots show that the sum from 0 to N of cos^2(2pifn) is approximately constant when
# f is not close to 0 or 1/2. 
# This means that the ML estimate can be simplified to
# f_hat = argmax the sum from 0 to N of - 2*x[n]-cos(2pifn)


# Plotting function to be maximized with 
N = 10
f_actual = 0.25 
fs = np.linspace(0,0.5,1000)
sigma_2 = 0.1
minfunc = []

def x_n(n):
    x = np.cos(2*np.pi*f_actual*n) + np.random.normal(0,np.sqrt(sigma_2))
    return x

plt.figure(2)

for j in range(5):
    xs = np.zeros(N)
    minfunc = np.zeros(len(fs))
    for n in range(N):
        xs[n] = x_n(n)

    for i in range(len(fs)):
        f = fs[i]
        for n in range(N):
            minfunc[i] += xs[n]*np.cos(2*np.pi*f*n)
    plt.plot(fs,minfunc, label = j)

plt.title("A few iterations of the functiont to be maximized")
plt.legend()

# As we can see the function clearly has a maximum around 0.25
deltaf = 0.001
fs = np.arange(0,0.5,deltaf)
current_max = np.zeros(5000)
maxf = np.zeros(5000)
for j in range(5000):
    xs = np.zeros(N)
    for n in range(N):
        xs[n] = x_n(n)
    for i in range(len(fs)):
        f = fs[i]
        current_sum = 0
        for n in range(N):
            current_sum +=  xs[n]*np.cos(2*np.pi*f*n)
            
        if  current_sum > current_max[j]:
            current_max[j] = current_sum
            maxf[j] = fs[i]


plt.figure(3)
plt.hist(maxf,100)


plt.show()
