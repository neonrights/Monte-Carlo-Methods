import random
import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from multiprocessing import Pool


def sufficient_statistic(grid):
	return float((grid[1:] != grid[:-1]).sum() + (grid[:,1:] != grid[:,:-1]).sum()) / (2 * grid.shape[0] * grid.shape[1])


def sample_site(origin, grid, beta):
	site_queue = deque()
	site_queue.append(origin)
	cluster = set()
	cluster.add(origin)

	while site_queue:
		site = site_queue.popleft()
		for i in [-1, 1]:
			neighbor = (site[0] + i, site[1])
			if neighbor not in cluster and 0 <= neighbor[0] < grid.shape[0]:
				if grid[neighbor] == grid[site] and random.random() < 1 - np.exp(-beta):
					cluster.add(neighbor)
					site_queue.append(neighbor)

			neighbor = (site[0], site[1] + i)
			if neighbor not in cluster and 0 <= neighbor[1] < grid.shape[1]:
				if grid[neighbor] == grid[site] and random.random() < 1 - np.exp(-beta):
					cluster.add(neighbor)
					site_queue.append(neighbor)

	label = random.random() < 0.5
	change = 0
	for pos in cluster:
		if label != grid[pos]:
			change += 1
		grid[pos] = label

	return grid, change


def sample_grid(grid, beta):
	remaining_sites = set([tuple(index) for index in np.ndindex(grid.shape)])
	new_grid = grid.copy()
	while remaining_sites:
		start = random.sample(remaining_sites, 1)[0]
		cluster = set([start])
		site_queue = deque([start])
		while site_queue:
			site = site_queue.popleft()
			for i in [-1, 1]:
				neighbor = (site[0] + i, site[1])
				if neighbor not in cluster and neighbor in remaining_sites and 0 <= neighbor[0] < grid.shape[0]:
					if grid[neighbor] == grid[site] and random.random() < 1 - np.exp(-beta):
						cluster.add(neighbor)
						site_queue.append(neighbor)

				neighbor = (site[0], site[1] + i)
				if neighbor not in cluster and neighbor in remaining_sites and 0 <= neighbor[1] < grid.shape[1]:
					if grid[neighbor] == grid[site] and random.random() < 1 - np.exp(-beta):
						cluster.add(neighbor)
						site_queue.append(neighbor)

		label = random.random() < 0.5
		for pos in cluster:
			new_grid[pos] = label

		remaining_sites -= cluster

	return new_grid, (new_grid != grid).sum()


def generate_single(beta, n=256, epsilon=1e-4, max_iters=10000000):
	indices = [tuple(index) for index in np.ndindex((n,n))]
	
	constant = np.ones((n,n), dtype=bool)
	checkered = np.zeros((n,n), dtype=bool)
	checkered[1::2,::2] = True
	checkered[::2,1::2] = True

	H_constant = list()
	H_checkered = list()
	H_constant.append(sufficient_statistic(constant))
	H_checkered.append(sufficient_statistic(checkered))

	CP_constant = list()
	CP_checkered = list()

	iters = 0
	print('\tstarted singles of %1.2f...' % beta)
	while np.abs(H_checkered[-1] - H_constant[-1]) > epsilon and iters < max_iters:
		site = indices[random.randint(0, n**2-1)]

		constant, constant_change = sample_site(site, constant, beta)
		checkered, checkered_change = sample_site(site, checkered, beta)

		CP_constant.append(constant_change)
		CP_checkered.append(checkered_change)
		
		H_constant.append(sufficient_statistic(constant))
		H_checkered.append(sufficient_statistic(checkered))

		iters += 1

	fig = plt.figure()
	plt.imshow(constant)
	plt.savefig('images/single %d constant %1.2f.png' % (n, beta))

	fig = plt.figure()
	plt.imshow(checkered)
	plt.savefig('images/single %d checkered %1.2f.png' % (n, beta))

	for i in range(1000):
		site = indices[random.randint(0, n**2-1)]

		constant, _ = sample_site(site, constant, beta)
		checkered, _ = sample_site(site, checkered, beta)

		H_constant.append(sufficient_statistic(constant))
		H_checkered.append(sufficient_statistic(checkered))

	fig = plt.figure()
	plt.plot(H_constant)
	plt.plot(H_checkered)
	plt.axvline(x=iters, color='r')
	plt.title("Sufficient Statistics, H, over Iteration for B=%1.2f" % beta)
	plt.xlabel("iteration")
	plt.ylabel("H")
	plt.savefig("images/single_H_beta_%1.2f.png" % beta)

	fig = plt.figure()
	plt.plot(CP_constant)
	plt.plot(CP_checkered)
	plt.title("CP per Sweep per Iteration for B=%1.2f" % beta)
	plt.xlabel("iteration")
	plt.ylabel("CPs")
	plt.savefig("images/single_CP_beta_%1.2f.png" % beta)

	print "single", beta, iters, float(sum(CP_constant)) / iters, float(sum(CP_checkered)) / iters, iters < max_iters
	return constant, checkered, iters


def generate_sweep(beta, n=256, epsilon=1e-4, max_iters=1000000):
	indices = [tuple(iters) for iters in np.ndindex((n,n))]
	
	constant = np.ones((n,n), dtype=bool)
	checkered = np.zeros((n,n), dtype=bool)
	checkered[1::2,::2] = True
	checkered[::2,1::2] = True

	H_constant = list()
	H_checkered = list()
	H_constant.append(sufficient_statistic(constant))
	H_checkered.append(sufficient_statistic(checkered))

	CP_constant = list()
	CP_checkered = list()

	iters = 0
	print('\tstarted sweeps of %1.2f...' % beta)
	while np.abs(H_checkered[-1] - H_constant[-1]) > epsilon and iters < max_iters:
		constant, constant_change = sample_grid(constant, beta)
		checkered, checkered_change = sample_grid(checkered, beta)

		H_constant.append(sufficient_statistic(constant))
		H_checkered.append(sufficient_statistic(checkered))
		CP_constant.append(constant_change)
		CP_checkered.append(checkered_change)

		iters += 1

	fig = plt.figure()
	plt.imshow(constant)
	plt.savefig('images/grid %d constant %1.2f.png' % (n, beta))

	fig = plt.figure()
	plt.imshow(checkered)
	plt.savefig('images/grid %d checkered %1.2f.png' % (n, beta))

	fig = plt.figure()
	plt.plot(H_constant)
	plt.plot(H_checkered)
	plt.axvline(x=iters, color='r')
	plt.title("Sufficient Statistics, H, over Iteration for B=%1.2f" % beta)
	plt.xlabel("iteration")
	plt.ylabel("H")
	plt.savefig("images/H_beta_%1.2f.png" % beta)

	fig = plt.figure()
	plt.plot(CP_constant)
	plt.plot(CP_checkered)
	plt.title("CP per Sweep per Iteration for B=%1.2f" % beta)
	plt.xlabel("iteration")
	plt.ylabel("CPs")
	plt.savefig("images/CP_beta_%1.2f.png" % beta)

	print "sweep", beta, iters, float(sum(CP_constant)) / iters, float(sum(CP_checkered)) / iters, iters < max_iters
	return constant, checkered, iters


if __name__ == '__main__':
	#betas = [0.65, 0.75, 0.85, 1.00]
	generate_single(1.0)
	"""
	for beta in betas:
		generate_single(beta)
		generate_sweep(beta)
	
	pool = Pool(4)
	pool.map_async(generate_single, betas)
	#pool.map_async(generate_sweep, betas)
	pool.close()
	pool.join()
	#"""