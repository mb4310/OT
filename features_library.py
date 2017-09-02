import numpy as np 

def first_moments():
	def q1(x):
		return x[0]

	def q2(x):
		return x[1]

	def q3(x):
		return x[2]

	def q11(x):
		return 1

	def q12(x):
		return 0

	def q13(x):
		return 0

	def q21(x):
		return 0

	def q22(x):
		return 1

	def q23(x):
		return 0

	def q31(x):
		return 0

	def q32(x):
		return 0

	def q33(x):
		return 1

	M = np.array([q1, q2, q3])
	J_M = np.array([[q11,q12,q13],[q21,q22, q23],[q31,q32,q33]])

	return (M,J_M)


def first_and_second_moments():
	def g1(x):
		return x[0]

	def g2(x):
		return x[1]

	def g3(x):
		return x[2]

	def g4(x):
		return x[0]*x[1]

	def g5(x): 
		return x[1]*x[2]

	def g6(x): 
		return x[0]*x[2]

	def g7(x):
		return x[0]**2

	def g8(x): 
		return x[1]**2

	def g9(x): 
		return x[2]**2

	def f11(x):
		return 1

	def f12(x):
		return 0

	def f13(x): 
		return 0

	def f21(x):
		return 0

	def f22(x):
		return 1

	def f23(x):
		return 0

	def f31(x):
		return 0

	def f32(x):
		return 0

	def f33(x):
		return 1

	def f41(x):
		return x[1]

	def f42(x):
		return x[0]

	def f43(x):
		return 0

	def f51(x):
		return 0

	def f52(x):
		return x[2]

	def f53(x):
		return x[1]

	def f61(x):
		return x[2]

	def f62(x):
		return 0

	def f63(x):
		return x[0]

	def f71(x):
		return 2*x[0]

	def f72(x):
		return 0

	def f73(x):
		return 0

	def f81(x):
		return 0

	def f82(x):
		return 2*x[1]

	def f83(x):
		return 0

	def f91(x):
		return 0

	def f92(x):
		return 0

	def f93(x):
		return 2*x[2]

	M = np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9])

	J_M = np.array([[f11,f12,f13],[f21,f22,f23],[f31,f32,f33],[f41,f42,f43],[f51,f52,f53],[f61,f62,f63],[f71,f72,f73],[f81,f82,f83],[f91,f92,f93]])

	return (M,J_M)



d = 3
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

grid = np.array(grid_generator(2,2,2))
bandwidths = []

for k in np.arange(grid.shape[0]):
	bandwidths.append(5.0)


def gaussian_pdf(x, mu, sigma):     
	return np.exp(-1 * np.linalg.norm(x-mu)**2/(2 * sigma **2))

def kernels_definer(i, locs, bandwidth):
	def ith_kernel(x):
		return gaussian_pdf(x, locs[i,:], bandwidth)
	return ith_kernel


def kernels_grad_definer(i,j, locs, bandwidth):
	def g_ij(x):
		return  -1 * (x[j]-locs[i][j])/(bandwidth**2) * gaussian_pdf(x, locs[i,:], bandwidth[i])
	return g_ij



G = np.empty(grid.shape[0], dtype = object)

J_G = np.empty((grid.shape[0], d), dtype = object)


for k in np.arange(grid.shape[0]):
	G[k] = kernels_definer(k)

for i in np.arange(grid.shape[0]):
	for j in np.arange(d):
		J_G[i,j] = kernels_grad_definer(i,j)