import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

N = 10**4
mu = 0
sigma = 1
lbda = 0.3

mus = np.ones(N)*mu
sigmas = np.ones(N)*sigma

x = np.random.normal(mus,sigmas)
y = np.linspace(0,N,N)

plt.figure()
plt.scatter(y[:100],x[:100],marker="x",label="random values")
lbdas = np.ones(N)*lbda
lin = np.linspace(-1,101,N)
plt.plot(lin,lbdas,color = "black",label="Threshold = 0.3")
plt.title("N random samples from a gaussian distribution")
plt.xlabel("n")
plt.ylabel("value")
plt.legend()


plt.figure()



plt.hist(x,bins=40,density=True,label="Data")
def gaussian_pdf(x,mu,sigma):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-((x-mu)**2)/(2*sigma**2))

x_pdf = np.linspace(-3,3,500)

y_pdf = gaussian_pdf(x_pdf,0,1)

plt.plot(x_pdf,y_pdf,label="pdf")
plt.legend()


plt.show()
