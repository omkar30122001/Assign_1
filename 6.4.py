import numpy as np
import matplotlib.pyplot as plt


def x(n):
    if n >= 0 and n < 6:
        x_dict = {0:1,1:2,2:3,3:4,4:2,5:1}
        return x_dict[n]
    else:
        return 0

def h(n):
    if n < 0:
        return 0

    elif n == 0:
        return 1    #h(0) = 1
        
    elif n == 1:
        return -0.5 #h(1) = -1/2
    else:
        return 5.0 * ((-0.5)**n)

n = 16
xn_i = [x(i) for i in range(n)]
hn_i = [h(i) for i in range(n)]

def fft(x):

    N = len(x)
    
    if N == 1:
        return x
    else:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        factor = np.exp(-2j*np.pi*np.arange(N)/ N)
        
        X = np.concatenate(\
            [X_even+factor[:int(N/2)]*X_odd,
             X_even+factor[int(N/2):]*X_odd])
        return X

Xo = fft(xn_i)
X = np.real(Xo)

Ho = fft(hn_i)
H = np.real(Ho)/n

Yo = []
for k in range(n):
    Yo.append(Xo[k] * Ho[k])
Y = np.real(Yo)

plt.figure(figsize=(12,10))
plt.subplot(2,1,1)
plt.stem(range(0,n),X)
plt.ylabel('$X(k)$')
plt.grid()
plt.title('X(k) using FFT')
plt.subplot(2,1,2)
plt.stem(range(0,n),H)
plt.ylabel('$H(k)$')
plt.grid()
plt.title('H(k) using FFT')
plt.savefig('6.4_FFT.pdf')
plt.show()

def ifft(X):

    Xx = np.conjugate(X)
    x = fft(Xx)
    return x
    
yn = ifft(Yo)
y = np.real(yn)

plt.figure(figsize=(12,10))
plt.subplot(2,1,1)
plt.stem(range(0,n),Y)
plt.title('Y(k)')
plt.ylabel('$Y(k)$')
plt.grid()
plt.subplot(2,1,2)
plt.stem(range(0,n),y)
plt.xlabel('$k$')
plt.ylabel('$y(k)$')
plt.title('y(n) using IFFT')
plt.grid()
plt.savefig('6.4_IFFT.pdf')
plt.show()