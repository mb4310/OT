import matplotlib.pyplot as plt 
import numpy as np
from SOT import sample_OT_solver
from Stoch_LOT import local_OT_solver
from new_LOT import compute_features


mu1 = [5,5]
cov1 = [[1,0],[0,1]]

mu4 = [20,8]
cov4 = [[1,0],[0,1]]

mu5 = [25,20]
cov5 = [[1,0],[0,1]]

mu2 = [7,7]
cov2 = [[1,0],[0,1]]

mu3 = [22,6]
cov3 = [[1,0],[0,1]]

mu6 = [27,22]
cov6 = [[1,0],[0,1]]


np.random.seed(42)
x = np.random.multivariate_normal(mu1, cov1, 1000)
x2 = np.random.multivariate_normal(mu4,cov4,1000)
x3 = np.random.multivariate_normal(mu5,cov5,1000)
y = np.random.multivariate_normal(mu2,cov2, 1000)
z = np.random.multivariate_normal(mu3,cov3, 1000)
w = np.random.multivariate_normal(mu6,cov6,1000)
Y = np.concatenate((y,z,w), axis=0)
X = np.concatenate((x,x2,x3), axis=0)

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

grid = 20 * np.array(grid_generator(3,2)) 

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

G = np.concatenate((M, ker1, ker2, ker3))
J_G = np.concatenate((J_M, J_ker1, J_ker2, J_ker3), axis=0)


plt.figure(1)
plt.plot(X[:,0], X[:,1], '.r')
plt.plot(Y[:,0], Y[:,1], '.b')
plt.plot(grid[:,0], grid[:,1], 'xg')
plt.plot(kernel_centers[:,0], kernel_centers[:,1], '.k')
plt.show()


A1 = compute_features(ker1, y)
A2 = compute_features(ker1, z)
A3 = compute_features(ker2, y)
A4 = compute_features(ker2, z)
A5 = compute_features(ker1, Y)
A6 = compute_features(ker2, Y)
print(compute_features(G,Y))
print(compute_features(ker3, Y))






s0 = np.zeros(M.shape[0])
(a,b,c) = local_OT_solver(M,J_M,X,Y, 0.000005, s0, 1)


fig = plt.figure()
plt.subplot(211)
plt.title('Optimal Transport Between Gaussian Distributions: First Attempt')
plt.plot(X[:,0], X[:,1], '.r', label='Initial Distribution')
plt.plot(Y[:,0], Y[:,1], '.b', label='Target Distribution')
# plt.plot(grid[:,0], grid[:,1], 'xg', label='Initial Kernel Grid')
# plt.plot(kernel_centers[:,0], kernel_centers[:,1], '.k', label='Kernel Centers')
plt.legend()




plt.subplot(212)
plt.plot(X[:,0], X[:,1], '.r', label='Initial Distribution')
plt.plot(c[:,0], c[:,1], '.k', label='Image Distribution')
# plt.plot(grid[:,0], grid[:,1], 'xg', label='Kernel Grid')
plt.legend()


fig.savefig('OT_Example2.png')

plt.show()