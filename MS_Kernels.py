import numpy as np
from mean_shift import mean_shift_cluster

def Q(x,h):      													
	return np.exp(-1 * np.linalg.norm(x)**2/(2 * h**2))


def grid_generator(a,b,c):   
	grid_centers = []
	bandwidths = []
	for i in np.arange(1, a):
		x = i/a
		for j in np.arange(1,b):
			y = j/b
			for k in np.arange(1, c):
				z = k  / c
		grid_centers.append((x,y,z))


	return grid_centers



def MS_Kernel_Definer(X, grid, bandwidth):
	def ker(x):
		return Q(x,bandwidth)

	new_grid = mean_shift_cluster(grid, X, Q, 10**(-2))
