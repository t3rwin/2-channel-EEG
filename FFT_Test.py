import numpy as np
import matplotlib.pyplot as plt

dt = 0.001
t = np.arange(0,1,dt)
f = np.sin(2*np.pi*50*t) + np.sin(2*np.pi*100*t)
f_clean = f
f = f + 2.5*np.random.randn(len(t))

n = len(t)
fhat = np.fft.fft(f,n)
cPSD = fhat*np.conj(fhat) / n
freq = (1/(dt*n)) * np.arange(n)
L = np.arange(1,np.floor(n/2), dtype = 'int')

fig, axs = plt.subplots(2,1)
plt.sca(axs[0])
plt.plot(t,f, color = 'c', linewidth = 0.5, label = 'noisy')
plt.plot(t,f_clean, color = 'k', linewidth = 0.5, label = 'clean')

plt.sca(axs[1])
plt.plot(freq[L], cPSD[L], color = 'c', linewidth = 0.5)

plt.show()