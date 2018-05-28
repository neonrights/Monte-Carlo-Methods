import random
import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
plt.switch_backend('agg')

import pdb


# higher chance of going up and right to hopefully end up in corner more often
def sample_corner_walk(n=10, epsilon=2., decay=0.8):
	def _corner_walk(state, position, weight, length, epsilon, path):
		total = 0.0
		options = []
		
		option = (position[0] - 1, position[1])
		if 0 <= position[0] - 1 <= n and state[option] == 0:
			total += 1.
			options.append(('l', option))

		option = (position[0] + 1, position[1])
		if 0 <= position[0] + 1 <= n and state[option] == 0:
			total += epsilon
			options.append(('r', option))

		option = (position[0], position[1] - 1)
		if 0 <= position[1] - 1 <= n and state[option] == 0:
			total += 1.
			options.append(('d', option))

		option = (position[0], position[1] + 1)
		if 0 <= position[1] + 1 <= n and state[option] == 0:
			total += epsilon
			options.append(('u', option))

		if total == 0:
			return weight, length, path

		prob = random.random()
		lower = 0.0
		for option in options:
			if option[0] == 'l' or option[0] == 'd':
				upper = 1. / total + lower
				if lower <= prob < upper:
					state[option[1]] = 1
					path.append(option[1])
					return _corner_walk(state, option[1], weight * total, length + 1, 1 + (epsilon - 1) * 0.9, path)

				lower = upper
			elif option[0] == 'r' or option[0] == 'u':
				upper = epsilon / total + lower
				if lower <= prob < upper:
					state[option[1]] = 1
					path.append(option[1])
					return _corner_walk(state, option[1], weight * total / epsilon, length + 1, 1 + (epsilon - 1) * 0.9, path)
				
				lower = upper


	init_state = np.zeros((n+1, n+1), dtype=np.uint8)
	init_state[0,0] = 1
	return _corner_walk(init_state, (0, 0), 1L, 0, epsilon, [(0,0)])


# recursive algorithm should seek to create as diverse samples as possible
def sample_branch_walk(n=10, branch=5, start=40):
	def _branch_walk(state, position, weight, path, split=True):
		options = list() # get potential options as list of positions
		for i in [-1, 1]:
			option = (position[0] + i, position[1])
			if 0 <= position[0] + i <= n and state[option] == 0:
				options.append(option)
			option = (position[0], position[1] + i)
			if 0 <= position[1] + i <= n and state[option] == 0:
				options.append(option)

		inv_prob = len(options)
		if inv_prob == 0: # no options, reached end of walk
			return weight, len(path), path

		if len(path) >= start and split:
			total = 0L
			lengths = 0
			max_len = 0
			max_path = None
			for i in xrange(branch):
				temp1, temp2, temp3 = _branch_walk(state.copy(), position, weight, path[:], False)
				total += temp1
				lengths += temp2
				if temp2 > max_len:
					max_len = temp2
					max_path = temp3

			return total / branch, lengths / branch, max_path
		else:
			choice = options[np.random.randint(0, inv_prob)]
			state[choice] = 1
			path.append(choice)
			return _branch_walk(state, choice, weight * inv_prob, path, split)

	# call helper with initial state
	init_state = np.zeros((n+1, n+1), dtype=np.uint8)
	init_state[0,0] = 1
	return _branch_walk(init_state, (0, 0), 1L, [(0,0)])


def sample_cutoff_walk(n=10, epsilon=0.05):
	def _cutoff_walk(state, position, weight, path):
		options = list() # get potential options as list of positions
		for i in [-1, 1]:
			option = (position[0] + i, position[1])
			if 0 <= position[0] + i <= n and state[option] == 0:
				options.append(option)
			option = (position[0], position[1] + i)
			if 0 <= position[1] + i <= n and state[option] == 0:
				options.append(option)

		inv_prob = len(options)
		if inv_prob == 0: # no options or early termination
			return weight, len(path), path
		if random.random() < epsilon:
			return weight / epsilon, len(path), path

		choice = options[np.random.randint(0, inv_prob)]
		state[choice] = 1
		path.append(choice)
		return _cutoff_walk(state, choice, weight * inv_prob / (1 - epsilon), path)

	# call helper with initial state
	init_state = np.zeros((n+1, n+1), dtype=np.uint8)
	init_state[0,0] = 1
	return _cutoff_walk(init_state, (0, 0), 1L, [(0,0)])


# returns weight and path taken
def sample_saw_walk(n=10):
	def _saw_walk(state, position, weight, path):
		options = list() # get potential options as list of positions
		for i in [-1, 1]:
			option = (position[0] + i, position[1])
			if 0 <= position[0] + i <= n and state[option] == 0:
				options.append(option)
			option = (position[0], position[1] + i)
			if 0 <= position[1] + i <= n and state[option] == 0:
				options.append(option)

		inv_prob = len(options)
		if inv_prob == 0: # no options, reached end of walk
			return weight, len(path), path

		choice = options[np.random.randint(0, inv_prob)]
		state[choice] = 1
		path.append(choice)
		return _saw_walk(state, choice, weight * inv_prob, path)

	# call helper with initial state
	init_state = np.zeros((n+1, n+1), dtype=np.uint8)
	init_state[0,0] = 1
	return _saw_walk(init_state, (0, 0), 1L, [(0,0)])


# process body for basic SAW call
def process_saw(sample_size, n=10):
	samples = [0L] * sample_size
	lengths = np.zeros(sample_size, dtype=int)
	max_path = None
	max_len = 0
	for i in xrange(sample_size):
		samples[i], lengths[i], path = sample_saw_walk(n)
		if lengths[i] > max_len:
			max_len = lengths[i]
			max_path = path

	return np.array(samples), lengths, max_path


def process_cutoff(sample_size, n=10, epsilon=0.1):
	samples = [0L] * sample_size
	lengths = np.zeros(sample_size, dtype=int)
	max_path = None
	max_len = 0
	for i in xrange(sample_size):
		samples[i], lengths[i], path = sample_cutoff_walk(n, epsilon)
		if lengths[i] > max_len:
			max_len = lengths[i]
			max_path = path

	return np.array(samples), lengths, max_path


def process_branch(sample_size, n=10, epsilon=0.1):
	samples = [0L] * sample_size
	lengths = np.zeros(sample_size, dtype=int)
	max_path = None
	max_len = 0
	for i in xrange(sample_size):
		samples[i], lengths[i], path = sample_branch_walk(n)
		if lengths[i] > max_len:
			max_len = lengths[i]
			max_path = path

	return np.array(samples), lengths, max_path


# process body for corner to corner sampling
def process_corner(sample_size, n=10):
	samples = []
	lengths = []
	max_path = None
	max_len = 0
	for i in xrange(sample_size):
		weight, length, path = sample_saw_walk(n)
		if path[-1] == (n,n):
			samples.append(weight)
			lengths.append(length)
			if length > max_len:
				max_len = length
				max_path = path
	
	return np.array(samples), np.array(lengths, dtype=int), max_path


def process_ur_corner(sample_size, n=10, epsilon=2.):
	samples = []
	lengths = []
	max_path = None
	max_len = 0
	for i in xrange(sample_size):
		weight, length, path = sample_corner_walk(n, epsilon)
		if path[-1] == (n,n):
			samples.append(weight)
			lengths.append(length)
			if length > max_len:
				max_len = length
				max_path = path
	
	return np.array(samples), np.array(lengths, dtype=int), max_path


def format_output(output):
	samples = []
	lengths = []
	path = None
	max_path = 0
	for pair in values:
		samples.append(pair[0])
		lengths.append(pair[1])
		if len(pair[2]) > max:
			max_path = len(pair[2])
			path = pair[2]

	return np.concatenate(saw_samples), np.concatenate(saw_lengths), path


def draw_path(path, title, filename):
	fig = plt.figure()
	ax = plt.axes()
	line_segments = LineCollection([(path[i], path[i+1]) for i in range(len(path) - 1)])
	ax.add_collection(line_segments, autolim=True)
	ax.autoscale_view()
	ax.set_title(title)
	plt.savefig(filename)


if __name__ == '__main__':
	# initialize pool of workers
	sample_size = 10000000
	process_count = 4
	pool = multiprocessing.Pool(processes=process_count)
	args = [sample_size / process_count] * process_count
	
	'''# perform vanilla SAW sampling
	t0 = time.time()
	output = pool.map(process_saw, args)
	t1 = time.time()

	# process output
	saw_samples, saw_lengths, saw_path = format_output(output)
	
	print "Vanilla"
	print "estimate:\t%e\tlength:\t%f\ttime:\t%fs" % (saw_samples.sum() / sample_size, saw_lengths.mean(), t1 - t0)

	# perform cutoff sampling
	t0 = time.time()
	values = pool.map(process_cutoff, args)
	t1 = time.time()

	cutoff_samples = []
	cutoff_lengths = []
	cutoff_path = None
	cutoff_max = 0
	for pair in values:
		cutoff_samples.append(pair[0])
		cutoff_lengths.append(pair[1])
		if len(pair[2]) > cutoff_max:
			cutoff_max = len(pair[2])
			cutoff_path = pair[2]

	cutoff_samples = np.concatenate(cutoff_samples)
	cutoff_lengths = np.concatenate(cutoff_lengths)

	print "Cutoff"
	print "estimate:\t%e\tlength:\t%f\ttime:\t%fs" % (cutoff_samples.sum() / sample_size, cutoff_lengths.mean(), t1 - t0)

	# perform cutoff sampling
	t0 = time.time()
	values = pool.map(process_branch, args)
	t1 = time.time()

	branch_samples = []
	branch_lengths = []
	branch_path = None
	branch_max = 0
	for pair in values:
		branch_samples.append(pair[0])
		branch_lengths.append(pair[1])
		if len(pair[2]) > branch_max:
			branch_max = len(pair[2])
			branch_path = pair[2]

	branch_samples = np.concatenate(branch_samples)
	branch_lengths = np.concatenate(branch_lengths)

	print "Branch"
	print "estimate:\t%e\tlength:\t%f\ttime:\t%fs" % (branch_samples.sum() / sample_size, branch_lengths.mean(), t1 - t0)
	'''

	# perform vanilla corner sampling
	t0 = time.time()
	values = pool.map(process_corner, args)
	t1 = time.time()

	corner_samples = []
	corner_lengths = []
	corner_path = None
	corner_max = 0
	for pair in values:
		corner_samples.append(pair[0])
		corner_lengths.append(pair[1])
		if len(pair[2]) > corner_max:
			corner_max = len(pair[2])
			corner_path = pair[2]

	corner_samples = np.concatenate(corner_samples)
	corner_lengths = np.concatenate(corner_lengths)

	print "Corner Vanilla"
	print "estimate:\t%e\tlength:\t%f\ttime:\t%fs" % (corner_samples.sum() / sample_size, corner_lengths.mean(), t1 - t0)


	# perform vanilla corner sampling
	t0 = time.time()
	values = pool.map(process_ur_corner, args)
	t1 = time.time()

	corner_ur_samples = []
	corner_ur_lengths = []
	corner_ur_path = None
	corner_ur_max = 0
	for pair in values:
		corner_ur_samples.append(pair[0])
		corner_ur_lengths.append(pair[1])
		if len(pair[2]) > corner_ur_max:
			corner_ur_max = len(pair[2])
			corner_ur_path = pair[2]

	corner_ur_samples = np.concatenate(corner_ur_samples)
	corner_ur_lengths = np.concatenate(corner_ur_lengths)

	print "Corner Up-Right"
	print "estimate:\t%e\tlength:\t%f\ttime:\t%fs" % (corner_ur_samples.sum() / sample_size, corner_ur_lengths.mean(), t1 - t0)

	'''
	# compute sampled values per sample size
	ns = [1e0, 5e0, 1e1, 5e1, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7]
	saw = [saw_samples[:int(n)].mean() for n in ns]
	cutoff = [cutoff_samples[:int(n)].mean() for n in ns]
	branch = [branch_samples[:int(n)].mean() for n in ns]

	# plot saw convergence results
	fig1 = plt.figure()
	a, = plt.plot(ns, saw)
	b, = plt.plot(ns, cutoff)
	c, = plt.plot(ns, branch)
	plt.title("Estimated Number of SAWs over Sample Size per Design")
	plt.legend([a, b, c], ["Design 1", "Design 2", "Design 3"])
	plt.xscale('log')
	plt.yscale('log')
	plt.xlabel("Sample size")
	plt.ylabel("Estimated number of SAWs")
	plt.savefig("saw.png")

	# plot histogram of lengths
	saw_bins = np.zeros(200, dtype=int)
	cutoff_bins = np.zeros(200, dtype=int)
	branch_bins = np.zeros(200, dtype=int)

	for i in xrange(sample_size):
		saw_bins[saw_lengths[i]] += 1
		cutoff_bins[cutoff_lengths[i]] += 1
		branch_bins[branch_lengths[i]] += 1

	fig2 = plt.figure()
	a, = plt.plot(saw_bins)
	b, = plt.plot(cutoff_bins)
	c, = plt.plot(branch_bins)
	plt.title("Count of SAW Lengths per Design")
	plt.legend([a, b, c], ["Original", "Cutoff", "Branch"])
	plt.xlabel("Length")
	plt.ylabel("Counts")
	plt.savefig("length.png")'''

	# plot histogram of corner to corner walk lengths
	corner_bins = np.zeros(200, dtype=int)
	for length in corner_lengths:
		corner_bins[length] += 1

	corner_ur_bins = np.zeros(200, dtype=int)
	for length in corner_ur_lengths:
		corner_ur_bins[length] += 1

	fig3 = plt.figure()
	a, = plt.plot(corner_bins)
	b, = plt.plot(corner_ur_bins)
	plt.title("Count of Corner to Corner Lengths per Design")
	plt.legend([a, b], ["Original", "Up-Right"])
	plt.xlabel("Length")
	plt.ylabel("Counts")
	plt.savefig("corner_length.png")

	# draw SAW paths
	#draw_path(saw_path, "Longest SAW for Original Design", "original_saw.png")
	#draw_path(cutoff_path, "Longest SAW for Cutoff Design", "cutoff_saw.png")
	#draw_path(branch_path, "Longest SAW for Branch Design", "branch_saw.png")
	draw_path(corner_path, "Longest Corner to Corner SAW for Original Design", "original_corner.png")
	draw_path(corner_ur_path, "Longest Corner to Corner SAW for Up-Right Design", "upright_corner.png")