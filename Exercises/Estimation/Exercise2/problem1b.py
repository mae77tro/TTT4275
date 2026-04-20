import numpy as np
import matplotlib.pyplot as plt

# Importing t and x
t = np.loadtxt("t.txt",dtype=float)
x = np.loadtxt("x.txt",dtype=float)

N = len(t)
#Calculating H and A
H = np.column_stack((np.ones(100),t,np.sin(2*np.pi*t)))
#A = np.dot(np.linalg.inv(np.dot(np.linalg.matrix_transpose(H),H)),np.linalg.matrix_transpose(H))

# Finding (A,B,C) 
# Found out that using np.linalg.lstsq is better than calculating the actual moore-penrose pseudoinverse
Theta_hat = np.linalg.lstsq(H,x,rcond=None)[0]

[A,B,C] = Theta_hat

print(Theta_hat)

print("A = " + str(A))
print("B = " + str(B))
print("C = " + str(C))

X = np.dot(H,Theta_hat)
sigma_2 = 1

Fischer = np.dot(np.linalg.matrix_transpose(H),H)/sigma_2

CRLB = np.linalg.inv(Fischer)

print("CRLB A = " + str(CRLB[0][0]))
print("CRLB B = " + str(CRLB[1][1]))
print("CRLB C = " + str(CRLB[2][2]))

plt.plot(t,x,label = "Actual x")
plt.plot(t,X, label = "Estimated x")
plt.legend()
plt.xlabel("t")
plt.ylabel("x")
plt.show()