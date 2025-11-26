# import numpy as np
# import matplotlib.pyplot as plt
#
# xs = np.array([-1., 0., 4., 5., 6.])
#
# theta = np.array([0.5,0.5,6,7,1,4])
#
# plt.figure()
# plt.scatter(xs, np.zeros_like(xs))
# plt.yticks([])
# plt.xlabel("x")
# plt.title("Data points for the 1D Gaussian mixture example")
# plt.grid(True)
# plt.show()



import numpy as np
from scipy.stats import norm

x = np.array([-1, 0, 4, 5, 6])
pi1, pi2 = 0.5, 0.5
mu1, sigma1 = 6, 1
mu2, sigma2 = 7, 2  # sigma = sqrt(4) = 2

p = pi1 * norm.pdf(x, mu1, sigma1) + pi2 * norm.pdf(x, mu2, sigma2)
logL = np.sum(np.log(p))
print(round(logL, 1))  # outputs -44.7