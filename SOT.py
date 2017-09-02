import numpy as np 
from numpy import random
from numpy import linalg
from Stoch_LOT import local_OT_solver

#Given one has a local OT solver L(X,Y) which returns the optimal map f from an array of samples X to an array of samples Y...

def sample_OT_solver(F, J_F, X, Y, K):

	# Initialization
	gnit = 0
	lit = 0 
	N_x = X.shape[0] 	#Number of samples from X
	N_y = Y.shape[0] 	#Number of samples from Y
	M = F.shape[0]
	Z = np.empty([X.shape[0], X.shape[1]]) 
	X_0=np.copy(X)
	X_K=np.copy(Y)
	x_0 = np.zeros(M)
	X_all = [X_0]
	delta = 10**(-2)
	done = False


	for i in np.arange(N_x):
		j = random.randint(0,N_y)
		Z[i,:] = Y[j,:]

	def X(k): 
		return ((K-k)/K)*X_0+(k/K)*Z

	for k in range(1,K):
		X_all.append(X(k))

	X_all.append(X_K)
	
	Z_i = np.copy(X_0)

	#Compute local solutions

	while done == False: #stopping criterion
		gnit += 1
		Z_before = np.copy(Z_i)
		Z_i = np.copy(X_0)
		local_maps = []

		for k in np.arange(1,K+1):
			lit += 1 
			(s, local_opt_map, image_pts) = local_OT_solver(F, J_F, Z_i, X_all[k], delta, x_0,5)
			print('Local iteration ' + str(lit) + " completed.")
			Z_i = np.copy(image_pts)

			local_maps.append(local_opt_map)



		Z_after = np.copy(Z_i)

		for k in np.arange(1,K):		#update your X_k
			X_all[k] = ((K-k)/K)*X_0+(k/K)*Z_i

		
		Z_error = Z_after - Z_before 		#compute how much your map has changed
		changes = linalg.norm(Z_error, ord=2, axis=1)
		epsilon = np.sum(changes)
		print("Global iteration " + str(gnit) + " completed.")
		print("Global change: " + str(epsilon))
		done = (epsilon <= 25) or gnit >= 10


	#Return your final map! It's the composition of f_K(f_(K-1)...f_1) given by the last iteration of the while loop. 

	return (Z_i, local_maps)
