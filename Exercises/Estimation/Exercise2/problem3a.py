import numpy as np
import matplotlib.pyplot as plt

def p(theta,epsilon,x):
    n1 = epsilon/np.sqrt(2*np.pi)*np.exp(-0.5*(theta-x)**2)
    n2 = (1-epsilon)/np.sqrt(2*np.pi)*np.exp(-0.5*(theta+x)**2)
    return n1 + n2


xs = [3]
epsilons = [1/2,3/4]

theta = np.linspace(-5,5,1000)
pdf = {}

for epsilon in epsilons:
    pdf[epsilon] = {}
    for x in xs:
        pdf[epsilon][x] = p(theta,epsilon,x)
print(pdf[1/2])
plt.figure(1)
for epsilon in epsilons:
    for x in xs:
        plt.plot(theta,pdf[epsilon][x],label=f"$epsilon$ = {str(epsilon)}, $x = ${str(x)}")
plt.legend()
plt.ylabel("$p(\\theta|x)$")
plt.xlabel("$\\theta$")
plt.show()