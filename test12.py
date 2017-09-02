import numpy as np
import matplotlib.pyplot as plt 
from Stoch_LOT import local_OT_solver

mu1 = np.array([0,0])
cov1 = np.array([[1,0],[0,1]])

mu2 = np.array([10,0])
cov2 = np.array([[1,0],[0,1]])

mu3 = np.array([20,0])
cov3 = np.array([[1,0],[0,1]])

mu4 = np.array([2,2])
cov4 = np.array([[1,0],[0,1]])

mu5 = np.array([12,2])
cov5 = np.array([[1,0],[0,1]])

mu6 = np.array([22,2])
cov6 = np.array([[1,0],[0,1]])

mu_shift = [0,0]
cov_shift = [[0.2, 0], [0,0.2]]


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

def grid_generator(a,b):   
	grid_centers = []
	bandwidths = []
	for i in np.arange(1, a):
		x = i/a
		for j in np.arange(1,b):
			y = j/b
			grid_centers.append([x,y])


	return grid_centers

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

def distance_matrix(S):
	if S.ndim == 2:
		D = np.empty((S.shape[0],S.shape[0]))
		for i in np.arange(S.shape[0]):
			for j in np.arange(S.shape[0]):
				D[i,j]=np.linalg.norm((S[i,:]-S[j,:]), ord=2, axis=0)

	else: 
		D = np.empty((S.shape[0],S.shape[0]))
		for i in np.arange(S.shape[0]):
			for j in np.arange(S.shape[0]):
				D[i,j]=np.linalg.norm((S[i]-S[j]))

	return D

def remove_redundant(S, epsilon):
	D = distance_matrix(S)
	indices = []
	for i in np.arange(D.shape[0]):
		for j in np.arange(i+1,D.shape[1]):
			if D[i,j] < epsilon:
				indices.append(j)
	if S.ndim == 1:
		S_new = np.delete(S, indices)
	else:
		S_new = np.delete(S, indices, axis=0)
	return S_new

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


np.random.seed(55)

x1 = np.random.multivariate_normal(mu1,cov1,1500)
x2 = np.random.multivariate_normal(mu2,cov2, 1500)
x3 = np.random.multivariate_normal(mu3,cov3, 1500)

y1 = np.random.multivariate_normal(mu4,cov4, 5000)
y2 = np.random.multivariate_normal(mu5,cov5, 5000)
y3 = np.random.multivariate_normal(mu6,cov6, 5000)

shifts = np.random.multivariate_normal(mu_shift, cov_shift, 9)
print("shifts are: " + str(shifts))

X = np.concatenate((x1,x2,x3), axis=0)
Y = np.concatenate((y1,y2,y3), axis=0)

grid = [45,10] * np.array(grid_generator(4,4)) - [12,3]
kcs = mean_shift_cluster(grid, Y, K, 10**(-1), 1.5)
print(kcs.shape)

kcs_new = remove_redundant(kcs, 10**(-1))
print(kcs_new.shape)

kcs_shift = kcs + shifts

plt.figure(1)
plt.plot(X[:,0], X[:,1], '.r')
plt.plot(Y[:,0], Y[:,1], '.b')
plt.plot(grid[:,0], grid[:,1], 'xg')
plt.plot(kcs_shift[:,0], kcs_shift[:,1], '.k')
plt.show()


























