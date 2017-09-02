import matplotlib.pyplot as plt 
import numpy as np
from SOT import sample_OT_solver
from Stoch_LOT import local_OT_solver
from new_LOT import compute_features


mu1 = [18,18]
cov1 = [[1,0],[0,1]]

mu4 = [5,18]
cov4 = [[1,0],[0,1]]

mu5 = [10,31]
cov5 = [[2,0],[0,2]]

mu6 = [12,34]
cov6 = [[2,0],[0,2]]

mu2 = [20,20]
cov2 = [[1,0],[0,1]]

mu3 = [5,20]
cov3 = [[1,0],[0,1]]


np.random.seed(42)
x = np.random.multivariate_normal(mu1, cov1, 1000)
x2 = np.random.multivariate_normal(mu4,cov4,1000)
x3 = np.random.multivariate_normal(mu5,cov5,2500)
y = np.random.multivariate_normal(mu2,cov2, 1000)
z = np.random.multivariate_normal(mu3,cov3, 1000)
z2 = np.random.multivariate_normal(mu6,cov6,2500)
Y = np.concatenate((y,z, z2), axis=0)
X = np.concatenate((x,x2, x3), axis=0)

def first_moments():
	def g1(x):
		return x[0]

	def g2(x):
		return x[1]

	def f11(x):
		return 1

	def f12(x):
		return 0

	def f21(x):
		return 0

	def f22(x):
		return 1

	M = np.array([g1, g2])
	J_M = np.array([[f11, f12],[f21,f22]])
	return (M,J_M)

def first_and_second_moments():
	def g1(x):
		return x[0]

	def g2(x):
		return x[1]

	def g3(x):
		return x[0]*x[1]

	def g4(x):
		return x[0]**2

	def g5(x): 
		return x[1]**2

	def f11(x):
		return 1

	def f12(x):
		return 0

	def f21(x):
		return 0

	def f22(x):
		return 1

	def f31(x):
		return x[1]

	def f32(x):
		return x[0]

	def f41(x):
		return 2*x[0]

	def f42(x):
		return 0

	def f51(x):
		return 0

	def f52(x):
		return 2*x[1]



	M = np.array([g1, g2, g3, g4, g5])

	J_M = np.array([[f11,f12],[f21,f22],[f31,f32],[f41,f42],[f51,f52]])

	return (M,J_M) 

def grid_generator(a,b):   
	grid_centers = []
	bandwidths = []
	for i in np.arange(1, a):
		x = i/a
		for j in np.arange(1,b):
			y = j/b
			grid_centers.append([x,y])


	return grid_centers

grid = np.array([[12,34],[5,20],[20,20]])

def K(x,h):
	z = np.exp(-1 * np.linalg.norm(x)**2/(2 * h**2))
	return z

def kernel_mean(s, X, K, h): 
	N_x = X.shape[0]
	x = 0
	y = 0
	L = s - X
	for i in np.arange(N_x):
		x+=K(L[i,:], h)*X[i,:]
	for i in np.arange(N_x):
		y+=K(L[i,:], h)
	return x/y

def mean_shift_cluster(grid, X, K, epsilon, h):
	G = np.copy(grid)
	done = False
	nit = 0
	N_G = G.shape[0]
	while done == False:
		nit +=1 
		new_centers = np.empty(G.shape)
		for i in np.arange(N_G):
			new_centers[i,:] = kernel_mean(G[i,:], X, K, h)
		change = np.sum(np.linalg.norm((new_centers - G), ord=2, axis=1))
		G = np.copy(new_centers)
		done = (change <= epsilon) or (nit >= 25)
		print("Mean-shift iteration " + str(nit) + " completed, " + "epsilon = " + str(change))
	return G

kernel_centers = mean_shift_cluster(grid, Y, K, 10**(-1), 2.0)

def distance_matrix(S):
	D = np.empty((S.shape[0],S.shape[0]))
	for i in np.arange(S.shape[0]):
		for j in np.arange(S.shape[0]):
			D[i,j]=np.linalg.norm((S[i,:]-S[j,:]), ord=2, axis=0)

	return D

D = distance_matrix(kernel_centers)
print("D = " + str(D))

def gaussian_pdf(x, mu, sigma):  
	if np.linalg.norm(x-mu) <= 3*sigma:
	   	z = 50/(np.sqrt((2 * np.pi)**2)) * np.exp(-1 * np.linalg.norm(x-mu)**2/(2 * sigma **2))
	else: 
		z = 0
	return z

def kernels_definer(i, locs, bandwidth):
	def ith_kernel(x):
		return gaussian_pdf(x, locs[i,:], bandwidth)
	return ith_kernel

def kernels_grad_definer(i,j, locs, bandwidth):
	def g_ij(x):
		return  -1 * (x[j]-locs[i][j])/(bandwidth**2) * gaussian_pdf(x, locs[i,:], bandwidth)
	return g_ij


ker1 = np.empty(kernel_centers.shape[0], dtype = object)

J_ker1 = np.empty((kernel_centers.shape[0], 2), dtype = object)


for k in np.arange(kernel_centers.shape[0]):
	ker1[k] = kernels_definer(k, kernel_centers, 1.0)

for i in np.arange(kernel_centers.shape[0]):
	for j in np.arange(2):
		J_ker1[i,j] = kernels_grad_definer(i,j, kernel_centers, 1.0)

ker2 = np.empty(kernel_centers.shape[0], dtype = object)

J_ker2 = np.empty((kernel_centers.shape[0], 2), dtype = object)


for k in np.arange(kernel_centers.shape[0]):
	ker2[k] = kernels_definer(k, kernel_centers, 2.0)

for i in np.arange(kernel_centers.shape[0]):
	for j in np.arange(2):
		J_ker2[i,j] = kernels_grad_definer(i,j, kernel_centers, 2.0)

grid = [45,30] * np.array(grid_generator(5,5)) + [-12,5]

ker3 = np.empty(grid.shape[0], dtype=object)
J_ker3 = np.empty((grid.shape[0], 2), dtype=object)

for k in np.arange(grid.shape[0]):
	ker3[k] = kernels_definer(k, grid, 1.0)

for i in np.arange(grid.shape[0]):
	for j in np.arange(2):
		J_ker3[i,j] = kernels_grad_definer(i,j, grid, 1.0)

(M, J_M) = first_and_second_moments()

G = np.concatenate((M, ker1, ker2))
J_G = np.concatenate((J_M, J_ker1, J_ker2), axis=0)


plt.figure(1)
plt.plot(x[:,0], x[:,1], '.r')
plt.plot(y[:,0], y[:,1], '.b')
plt.plot(z2[:,0], z2[:,1], '.b')
plt.plot(z[:,0], z[:,1], '.b')
plt.plot(x2[:,0], x2[:,1], '.r')
plt.plot(x3[:,0], x3[:,1], '.r')
plt.plot(grid[:,0], grid[:,1], 'xg')
plt.plot(kernel_centers[:,0], kernel_centers[:,1], '.k')
plt.show()








s0 = np.zeros(G.shape[0])
(a,b,c) = local_OT_solver(G,J_G,X,Y, 0.3, s0, 1)




plt.figure(1)

plt.plot(c[:,0], c[:,1], '.k')
plt.plot(grid[:,0], grid[:,1], 'xg')
plt.plot(kernel_centers[:,0], kernel_centers[:,1], '.k')



plt.figure(2)
plt.plot(x[:,0], x[:,1], '.r', alpha=0.1)
plt.plot(x2[:,0], x2[:,1], '.r', alpha=0.1)
plt.plot(x3[:,0], x3[:,1], '.r', alpha=0.1)
plt.plot(c[:,0], c[:,1], '.k')
plt.plot(y[:,0], y[:,1], '.b', alpha=0.1)
plt.plot(z[:,0], z[:,1], '.b', alpha=0.1)
plt.plot(z2[:,0], z2[:,1], '.b', alpha=0.1)
plt.plot(grid[:,0], grid[:,1], 'xy')
plt.plot(kernel_centers[:,0], kernel_centers[:,1], 'xg')

plt.figure(3)
plt.plot(y[:,0], y[:,1], '.b', alpha=0.1)
plt.plot(z[:,0], z[:,1], '.b', alpha=0.1)
plt.plot(z2[:,0], z2[:,1], '.b', alpha=0.1)
plt.plot(kernel_centers[:,0], kernel_centers[:,1], 'xg')
plt.show()