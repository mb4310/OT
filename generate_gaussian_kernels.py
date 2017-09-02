import numpy as np

def grid_generator(a,b,c,A):   #A is the maximum intensity of the pixels
	grid_centers = []
	bandwidths = []
	for i in np.arange(1, a):
		x = i/a
		grid_centers.append(1, b)
		for j in np.arange(b):
			y = j/b
			for k in np.arange(1, c):
				z = k * A / c
		grid_centers.append(x,y,z)


	return grid_centers

grid = np.array(grid_generator(2,2,2))
bandwidths = []

for k in np.arange(grid.shape[0]):
	bandwidths.append(5.0)


def gaussian_pdf(x, mu, sigma, d):      ###CHECK THIS WORKS LOL
	return 1/(np.sqrt((2 * np.pi)**d)) * np.exp(-1 * np.linalg.norm(x-mu)**2/(2 * sigma **2))

def kernels_definer(i):
	def ith_kernel(x):
		return gaussian_pdf(x, grid[i,:], bandwidths[i], 3)
	return ith_kernel


def kernels_grad_definer(i,j):
	def g_ij(x):
		return  -1 * (x[j]-grid[i][j])/(bandwidths[i]**2) * gaussian_pdf(x, grid[i,:], bandwidths[i], 3)
	return g_ij



G = np.empty(grid.shape[0], dtype = object)

J_G = np.empty((grid.shape[0], d), dtype = object)


for k in np.arange(grid.shape[0]):
	G[k] = kernels_definer(k)

for i in np.arange(grid.shape[0]):
	for j in np.arange(d):
		J_G[i,j] = kernels_grad_definer(i,j)




