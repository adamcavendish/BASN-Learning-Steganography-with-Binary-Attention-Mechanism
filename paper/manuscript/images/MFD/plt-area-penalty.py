import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(0.001, 1, 100000)
Y = 1/2 * (1.1 * X)**(8 * X - 0.1)

plt.figure(figsize=(5, 5))
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.plot(X, Y)
plt.savefig('MFD-Area-Penalty.eps', format='eps')
plt.savefig('MFD-Area-Penalty.png', format='png')

