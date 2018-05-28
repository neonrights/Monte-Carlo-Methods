import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
import pdb


def metropolis_hastings(x, Y, eps, K, title, mu, sigma):
	U = lambda theta : np.matmul(theta, theta) / 2 + ((Y - theta[0] - theta[1]**2)**2).sum() / 8
	proposal = multivariate_normal([0., 0.], [[eps, 0.],[0., eps]])

	reject = 0
	print "burn in %s" % title
	pos = np.array(x)
	burnin = np.empty((1001,2))
	burnin[0] = pos
	for i in range(1000):
		for j in range(K):
			# single iteration
			last_pos = np.empty_like(pos)
			last_pos[:] = pos
			while (pos == last_pos).all():
				offset = proposal.rvs()
				pos += offset

				alpha = U(last_pos) - U(pos)
				alpha = np.exp(min(0, alpha))
				if random.random() >= alpha:
					# reject state, try again
					pos[:] = last_pos
					reject += 1

		burnin[i+1] = pos

	print "sampling %s" % title
	samples = np.empty((10000, 2))
	for i in range(10000):
		for j in range(K):
			# single iteration
			last_pos = np.empty_like(pos)
			last_pos[:] = pos
			while (pos == last_pos).all():
				offset = proposal.rvs()
				pos += offset

				alpha = U(last_pos) - U(pos)
				alpha = np.exp(min(0, alpha))
				if random.random() >= alpha:
					# reject state, try again
					pos[:] = last_pos
					reject += 1

		samples[i] = pos

	print float(reject) / (11000 * K)

	# visualize burn-in
	fig = plt.figure()
	plt.plot(burnin[:,0], burnin[:,1], '.r-')
	plt.title('%s Burn-in' % title)
	fig.savefig('%s burnin.png' % title)

	# plot samples over iterations
	fig = plt.figure()
	plt.plot(samples[:,0], samples[:,1], '.b')
	plt.title('%s Samples' % title)
	fig.savefig('%s samples.png' % title)

	# plot samples over iterations
	fig = plt.figure()
	plt.plot(samples[:,0])
	plt.title('%s X Samples per Iteration' % title)
	fig.savefig('%s X samples iters.png' % title)

	fig = plt.figure()
	plt.plot(samples[:,1])
	plt.title('%s Y Samples per Iteration' % title)
	fig.savefig('%s Y samples iters.png' % title)

	# calculate ESS
	var1, var2 = 0., 0.
	for i in range(10000-1):
		auto1, auto2 = 0., 0.
		for j in range(i+1, 10000):
			auto1 += samples[j, 1] * samples[10000 - j - 1, 0]
			auto2 += samples[j, 1] * samples[10000 - j - 1, 1]
		auto1 /= (10000 - i - 1) * sigma[0]
		auto2 /= (10000 - i - 1) * sigma[1]


	print "%s ESS X: %f" % (title, 10000. / (1 + 2 * var1))
	print "%s ESS Y: %f" % (title, 10000. / (1 + 2 * var1))


def hmc_sampler(x, Y, eps, L, K, title, mu=None, sigma=None):
	U = lambda theta : np.matmul(theta, theta) / 2 + ((Y - theta[0] - theta[1]**2)**2).sum() / 8
	U_grad = lambda theta : theta - (Y - theta[0] - theta[1]**2).sum() * np.array([1, 2*theta[1]]) / 4
	p = multivariate_normal([0., 0.], [[1., 0.],[0., 1.]])
	def iteration(position, momentum):
		momentum -= eps * U_grad(position) / 2
		if L > 1:
			for leapfrog in range(L-1):
				# perform leapfrog updates
				position += eps * momentum
				momentum -= eps * U_grad(position)

		position += eps * momentum
		momentum -= eps * U_grad(position)
		return position, -momentum

	reject = 0
	print "burn in %s" % title
	pos = np.array(x)
	burnin = np.empty((1001,2))
	burnin[0] = pos
	for i in range(1000):
		for j in range(K):
			# single iteration
			last_pos = np.empty_like(pos)
			last_pos[:] = pos
			last_mom = p.rvs()
			while (pos == last_pos).all():
				pos, mom = iteration(pos, last_mom[:])

				alpha = U(last_pos) + np.matmul(last_mom, last_mom) / 2 - U(pos) - np.matmul(mom, mom) / 2
				alpha = np.exp(min(0, alpha))
				if random.random() >= alpha:
					# reject state, try again
					pos[:] = last_pos
					last_mom = p.rvs()
					reject += 1

		burnin[i+1] = pos

	print "sampling %s" % title
	samples = np.empty((10000, 2))
	for i in range(10000):
		for j in range(K):
			# single iteration
			last_pos = np.empty_like(pos)
			last_pos[:] = pos
			last_mom = p.rvs()
			while (pos == last_pos).all():
				pos, mom = iteration(pos, last_mom)

				alpha = U(last_pos) + np.matmul(last_mom, last_mom) / 2 - U(pos) - np.matmul(mom, mom) / 2
				alpha = np.exp(min(0, alpha))
				if random.random() >= alpha:
					# reject state, try again
					pos[:] = last_pos
					last_mom = p.rvs()
					reject += 1

		samples[i] = pos

	print float(reject) / (11000 * K)

	# visualize burn-in
	fig = plt.figure()
	plt.plot(burnin[:,0], burnin[:,1], '.r-')
	plt.title('%s Burn-in' % title)
	fig.savefig('%s burnin.png' % title)

	# plot samples over iterations
	fig = plt.figure()
	plt.plot(samples[:,0], samples[:,1], '.b')
	plt.title('%s Samples' % title)
	fig.savefig('%s samples.png' % title)

	# plot samples over iterations
	fig = plt.figure()
	plt.plot(samples[:,0])
	plt.title('%s X Samples per Iteration' % title)
	fig.savefig('%s X samples iters.png' % title)

	fig = plt.figure()
	plt.plot(samples[:,1])
	plt.title('%s Y Samples per Iteration' % title)
	fig.savefig('%s Y samples iters.png' % title)

	# calculate ESS
	if mu is None:
		mu = samples.mean(1)
	if sigma is None:
		sigma = samples.var(1)

	var1, var2 = 0., 0.
	for i in range(10000-1):
		auto1, auto2 = 0., 0.
		for j in range(i+1, 10000):
			auto1 += samples[j, 1] * samples[10000 - j - 1, 0]
			auto2 += samples[j, 1] * samples[10000 - j - 1, 1]
		auto1 /= (10000 - i - 1) * sigma[0]
		auto2 /= (10000 - i - 1) * sigma[1]

		var1 += (1 - float(j) / 10000) * auto1
		var2 += (1 - float(j) / 10000) * auto2


	print "%s ESS X: %f" % (title, 10000. / (1 + 2 * var1))
	print "%s ESS Y: %f" % (title, 10000. / (1 + 2 * var1))
	return samples.var(0), samples.mean(0)


if __name__ == '__main__':
	x, y = np.mgrid[-2:2:0.01, -2:2:0.01]
	pos = np.empty(x.shape + (2,))
	pos[:,:,0] = x
	pos[:,:,1] = y

	Y = np.array([float(val) for val in open('project2_data.txt').read().split()])

	fuck = pos[:,:,0].reshape(pos.shape[:-1] + (1,))
	you = pos[:,:,1].reshape(pos.shape[:-1] + (1,))
	plt.contourf(x, y, (pos**2).sum(-1) / 2 + ((Y.reshape(1,1,-1) - fuck - you**2)**2).sum(-1) / 8)
	plt.title("Contour Plot of Energy of Banana Distribution")
	plt.savefig('banana_contour.png')

	x_init = [0.,0.]
	eps_ = 0.08

	sigma, mu = hmc_sampler(x_init, Y, eps_, 25, 1, "Banana Hamiltonian Monte-Carlo L=25")

	metropolis_hastings(x_init, Y, eps_, 25, "Banana Metropolis Hastings", mu, sigma)

	hmc_sampler(x_init, Y, eps_, 1, 1, "Banana Langevin Monte-Carlo K=1", mu, sigma)
	hmc_sampler(x_init, Y, eps_, 1, 25, "Banana Langevin Monte-Carlo K=25", mu, sigma)
	hmc_sampler(x_init, Y, eps_, 5, 1, "Banana Hamiltonian Monte-Carlo L=5", mu, sigma)
