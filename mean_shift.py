import numpy as np 
import scipy 

h = 1.0 			#Select your bandwidth, this is done by hand!
X =    				#your data is given as a numpy array!
N_x = X.shape[0]


def K(x,h):      													#Mean zero Gaussian kernel
	return np.exp(-1 * np.linalg.norm(x)**2/(2 * h**2))


def kernel_mean(s, X, K): 										#X is the set of data, s is the sample mean. This implementation is atrocious(!)
	N_x = X.shape[0]
	x = 0
	y = 0
	L = s - X
	for i in np.arange(N_x):
		x+=K(L[i,:])*X[i,:]
	for i in np.arange(N_x):
		y+=K(L[i,:])
	return x/y

def mean_shift_cluster(G, X, K, epsilon, h):						#G is an array of your initial cluster centers, epsilon is stopping threshold
	done = False
	nit = 0
	N_G = G.shape[0]
	while done == False:
		nit +=1 
		new_centers = np.empty(G.shape)
		for i in np.arange(N_G):
			new_centers[i,:] = kernel_mean(G[i,:], X, K)
		change = np.sum(np.linalg.norm((new_centers - G), ord=2, axis=1))
		G = np.copy(new_centers)
		done = (changes <= epsilon) or (nit >= 25)
		print("Mean-shift iteration " + str(nit) + " completed, " + "epsilon = " + str(change))
	return G





