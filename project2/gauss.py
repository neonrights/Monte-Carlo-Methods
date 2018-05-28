import random
import time
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
import pdb


def metropolis_hastings(x, pi, Q, K, title):
	pi_logprob = lambda a : -np.matmul(np.matmul(a, np.linalg.inv(pi.cov)), a) / 2
	Q_logprob = lambda a : -np.matmul(np.matmul(a, np.linalg.inv(Q.cov)), a) / 2

	reject = 0
	start = time.time()
	print "burn in %s" % title
	pos = np.array(x)
	burnin = np.empty((1001,2))
	burnin[0] = pos
	for i in range(1000):
		for j in range(K):
			last_pos = np.empty_like(pos)
			last_pos[:] = pos
			while (pos == last_pos).all():
				offset = Q.rvs()
				pos += offset
				alpha = pi_logprob(pos) - pi_logprob(last_pos)
				#pdb.set_trace()
				alpha = np.exp(min(0, alpha))
				if random.random() >= alpha:
					pos[:] = last_pos
					reject += 1

		burnin[i+1] = pos

	print float(reject) / (1000 * K)

	print "sampling %s" % title
	samples = np.empty((10000,2))
	for i in range(10000):
		for j in range(K):
			last_pos = np.empty_like(pos)
			last_pos[:] = pos
			while (pos == last_pos).all():
				offset = Q.rvs()
				pos += offset
				alpha = pi_logprob(pos) - pi_logprob(last_pos)
				alpha = np.exp(min(0, alpha))
				if random.random() >= alpha:
					pos[:] = last_pos

		samples[i] = pos

	end = time.time()
	print end - start


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
		auto1 /= (10000 - i - 1)
		auto2 /= (10000 - i - 1)

		var1 += (1 - float(j) / 10000) * auto1
		var2 += (1 - float(j) / 10000) * auto2


	print "%s ESS X: %f" % (title, 10000. / (1 + 2 * var1))
	print "%s ESS Y: %f" % (title, 10000. / (1 + 2 * var1))


def hmc_sampler(x, q, p, eps, L, K, title):
	inv_q = np.linalg.inv(q.cov)
	inv_p = np.linalg.inv(p.cov)
	def iteration(position, momentum):
		momentum -= eps * np.matmul(position, (inv_q + inv_q.T)) / 4
		if L > 1:
			for leapfrog in range(L-1):
				# perform leapfrog updates
				position += eps * np.matmul(inv_p, momentum)
				momentum -= eps * np.matmul(position, (inv_q + inv_q.T)) / 2

		position += eps * np.matmul(inv_p, momentum)
		momentum -= eps * np.matmul(position, (inv_q + inv_q.T)) / 4
		return position, -momentum

	reject = 0
	start = time.time()
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

				alpha = -np.matmul(np.matmul(pos, inv_q), pos) - np.matmul(np.matmul(mom, inv_p), mom)
				alpha += np.matmul(np.matmul(last_pos, inv_q), last_pos) + np.matmul(np.matmul(last_mom, inv_p), last_mom)
				alpha = np.exp(min(0, alpha / 2))
				if random.random() >= alpha:
					# reject state, try again
					pos[:] = last_pos
					last_mom = p.rvs()
					reject += 1

		burnin[i+1] = pos

	print float(reject) / (1000 * K)

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

				alpha = -np.matmul(np.matmul(pos, inv_q), pos) - np.matmul(np.matmul(mom, inv_p), mom)
				alpha += np.matmul(np.matmul(last_pos, inv_q), last_pos) + np.matmul(np.matmul(last_mom, inv_p), last_mom) 
				alpha = np.exp(min(0, alpha / 2))
				if random.random() >= alpha:
					# reject state, try again
					pos[:] = last_pos
					last_mom = p.rvs()

		samples[i] = pos

	end = time.time()
	print end - start

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
		auto1 /= (10000 - i - 1)
		auto2 /= (10000 - i - 1)

		var1 += (1 - float(j) / 10000) * auto1
		var2 += (1 - float(j) / 10000) * auto2


	print "%s ESS X: %f" % (title, 10000. / (1 + 2 * var1))
	print "%s ESS Y: %f" % (title, 10000. / (1 + 2 * var1))


if __name__ == '__main__':
	x, y = np.mgrid[-3:3:0.01, -3:3:0.01]
	pos = np.empty(x.shape + (2,))
	pos[:,:,0] = x
	pos[:,:,1] = y

	q_cov = np.array([[1., 0.998],[0.998, 1.]])
	mean = np.array([0., 0.])
	q = multivariate_normal(mean, q_cov)

	plt.contourf(x, y, -q.logpdf(pos))
	plt.title("Contour Plot of Energy of Non-Isotropic Gaussian Distribution")
	plt.savefig('gaussian_contour.png')

	init_x = [0., -10.]
	eigs, _ = np.linalg.eig(q_cov)
	eps_ = np.sqrt(eigs.min())
	leapfrog_ = int(np.sqrt(eigs.max()) / eps_ + 1)
	#pdb.set_trace()

	p1 = multivariate_normal(mean, [[1., 0.], [0., 1.]])
	p2 = multivariate_normal(mean, np.linalg.inv(q_cov))
	Q1 = multivariate_normal(mean, [[eps_, 0.], [0., eps_]])

	# direct samples
	true_samples = q.rvs(10000)
	fig = plt.figure()
	plt.plot(true_samples[:,0], true_samples[:,1], '.b')
	plt.title('True Samples')
	fig.savefig('direct samples.png')

	fig = plt.figure()
	plt.plot(true_samples[:,0])
	plt.title('True X Samples over Iteration')
	fig.savefig('direct X samples iter.png')

	fig = plt.figure()
	plt.plot(true_samples[:,1])
	plt.title('True Y Samples over Iteration')
	fig.savefig('direct Y samples iter.png')

	# hmc and lmc samples	
	hmc_sampler(init_x, q, p1, eps_, leapfrog_, 1, 'Isotropic-Gaussian Langevin Monte-Carlo')
	hmc_sampler(init_x, q, p1, eps_, leapfrog_/2, 1, 'Isotropic-Gaussian Hamiltonian Monte-Carlo with I2')
	hmc_sampler(init_x, q, p1, eps_, 1, leapfrog_, 'Isotropic-Gaussian Langevin Monte-Carlo with I2')
	hmc_sampler(init_x, q, p2, eps_, leapfrog_/2, 1, 'Isotropic-Gaussian Hamiltonian Monte-Carlo with Ideal')
	hmc_sampler(init_x, q, p2, eps_, 1, leapfrog_, 'Isotropic-Gaussian Langevin Monte-Carlo with Ideal')

	# metropolis-hastings with gaussian proposals
	metropolis_hastings(init_x, q, Q1, leapfrog_, "Isotropic-Gaussian Metropolis Hastings")
