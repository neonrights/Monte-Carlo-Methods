import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

samples1 = np.random.normal(loc=2.0, scale=1.0, size=(10000000, 2))
samples2 = np.random.normal(loc=0.0, scale=1.0, size=(10000000, 2))
samples3 = np.random.normal(loc=0.0, scale=4., size=(10000000, 2))
est1, est2, est3 = [], [], []
eff2, eff3 = [0], [0]
ns = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000]
for n in ns:
	samp1 = samples1[:n]
	samp2 = samples2[:n]
	samp3 = samples3[:n]

	mag1 = np.sqrt(np.sum(samp1**2, axis=1))
	mag2 = np.sqrt(np.sum(samp2**2, axis=1))
	mag3 = np.sum(samp3**2, axis=1)
	weight2 = np.exp(2 * samp2.sum(1) - 4)
	weight3 = 16 * np.exp(mag3 / 32 - np.sum((samp3 - 2)**2, axis=1) / 2)

	est1.append(np.mean(mag1, axis=0))
	est2.append(np.mean(mag2 * weight2, axis=0))
	est3.append(np.mean(np.sqrt(mag3) * weight3, axis=0))

	if n > 1:
		eff2.append(float(n) / (1 + np.var(weight2, ddof=1)))
		eff3.append(float(n) / (1 + np.var(weight3, ddof=1)))

fig1 = plt.figure()
a, = plt.plot(ns, est1)
b, = plt.plot(ns, est2)
c, = plt.plot(ns, est3)
plt.legend([a, b, c], [r"$\hat{\theta}_1$", r"$\hat{\theta}_2$", r"$\hat{\theta}_3$"])
plt.xscale('log')
plt.title(r"$\hat{\theta}_1$, $\hat{\theta}_2$, and $\hat{\theta}_3$ estimates over sample size")
plt.xlabel("Samples size")
plt.ylabel(r"$\hat{\theta}$ estimates")
plt.savefig("theta.png")

fig2 = plt.figure()
a, = plt.plot(ns, ns)
b, = plt.plot(ns, eff2)
c, = plt.plot(ns, eff3)
plt.legend([a, b, c], ["ground truth", r"$ess^*(n_2)$", r"$ess^*(n_3)$"])
plt.xscale('log')
plt.yscale('log')
plt.title("Effective sample size over true sample size")
plt.xlabel("Samples size")
plt.ylabel("Effective sample size")
plt.savefig("ess.png")