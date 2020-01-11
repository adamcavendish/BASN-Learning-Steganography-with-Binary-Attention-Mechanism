import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(0, 1)
Y = X**(3-2*X)

plt.figure(figsize=(5, 5))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.plot(X, Y)
plt.savefig('ITC-Area-Penalty.eps', format='eps')
plt.savefig('ITC-Area-Penalty.png', format='png')

