import random
import numpy as np
import matplotlib.pyplot as plt

from math import exp

import pdb

N = 64
SAMPLES = 100
MAX_SAMPLES = 10000000

def alter(X, site, r):
	a, b = 0.0, 0.0
	for offset in [-1, 1]:
		neighbor = (site[0] + offset, site[1])
		if 0 <= neighbor[0] < N:
			a += beta * (X[site] != X[neighbor])
			b += beta * (X[site] == X[neighbor])

		neighbor = (site[0], site[1] + offset)
		if 0 <= neighbor[1] < N:
			a += beta * (X[site] != X[neighbor])
			b += beta * (X[site] == X[neighbor])

	if X[site]:
		prob = exp(b) / (exp(a) + exp(b))
	else:
		prob = exp(a) / (exp(a) + exp(b))

	X[site] = r < prob


if __name__ == '__main__':
	indices = [tuple(index) for index in np.ndindex((N,N))]
	taus = list()
	betas = [0.5, 0.65, 0.75, 0.83, 0.84, 0.85, 0.9, 1.0]
	for beta in betas:
		X1 = np.ones((N,N), dtype=np.uint8)
		X2 = np.zeros((N,N), dtype=np.uint8)

		state1, state2 = np.empty(MAX_SAMPLES + SAMPLES + 1, dtype=np.uint64), np.empty(MAX_SAMPLES + SAMPLES + 1, dtype=np.uint64)
		state1[0] = X1.sum()
		state2[0] = X2.sum()
		print beta

		index = 1
		while not (X1 == X2).all() and index <= MAX_SAMPLES:
			site = indices[random.randint(0, N**2-1)]
			r = random.random()

			alter(X1, site, r)
			alter(X2, site, r)

			state1[index] = X1.sum()
			state2[index] = X2.sum()
			index += 1
		
		if index <= MAX_SAMPLES:
			print index
			taus.append(index)
			fig = plt.figure()
			plt.imshow(X1)
			plt.savefig('B=%1.2f Output.png' % beta)
 
			for step in range(SAMPLES):
				site = indices[random.randint(0, N**2-1)]
				r = random.random()

				alter(X1, site, r)
				alter(X2, site, r)

				state1[index] = X1.sum()
				state2[index] = X2.sum()
				index += 1

		fig = plt.figure()
		plt.plot(state1[:index])
		plt.plot(state2[:index])
		plt.title("Plot of Black and White Image Sum for B = %1.2f" % beta)
		plt.xlabel("Iteration")
		plt.ylabel("Sum of Image")
		plt.savefig("B=%1.2f.png" % beta)

	fig = plt.figure()
	plt.plot(betas[:len(taus)], taus)
	plt.title(r"Plot of \tau over \beta")
	plt.xlabel(r"\beta value")
	plt.ylabel(r"\tau value")
	plt.savefig("tau.png")

