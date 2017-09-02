import numpy as np
from scipy import misc

def compute_features(G, Y): #G as a ~2Darray~(shape Mx1) vector of feature functions
	
	M = G.shape[0]
	N_y = Y.shape[0]
	b = np.empty((M))
	for i in np.arange(M):
		k=0.00001
		S = np.empty((N_y))
		for j in np.arange(N_y):
			S[j] = G[i](Y[j,:])
			if S[j] > 0:
				k += 1
		b[i] = np.sum(S)/k

	return b


def evaluate_functions(G,x): #takes an array of functions g_{ij}=G_{i}{j} and a point 'x', returns a numpy array G(x)_{ij} = g_{ij}(x)
	if G.ndim == 2:

		Gx = np.empty(G.shape)
		for i in np.arange(G.shape[0]):
			for j in np.arange(G.shape[1]):
				Gx[i,j] = G[i,j](x)

		return Gx

	else:
		Gx = np.empty(G.shape)
		for i in np.arange(G.shape[0]):
			Gx[i] = G[i](x)

		return Gx




def compute_Ai(J_G,X):  
	N_x = X.shape[0]
	A = np.empty((N_x, J_G.shape[0], J_G.shape[1]))
	for i in np.arange(N_x):
		A_i = evaluate_functions(J_G, X[i,:])
		A[i,:,:] = A_i 
	return A



def local_OT_solver(G, J_G, X,Y,epsilon, s_0,B):

	M = G.shape[0]
	N_x = X.shape[0]
	d = X.shape[1]
	N_y = Y.shape[0]
	I = np.identity(M)
	b = compute_features(G,Y)
	A = compute_Ai(J_G,X)
	nit = 0
	bnit = 0
	delta = 0.3
	done = False
	x = np.copy(s_0)
	N_B = N_x // B


	def Z(s): 
		Z = np.empty((X.shape[0],X.shape[1]))
		for i in np.arange(N_x):
			Z[i,:] = X[i,:] + np.dot(s,A[i,:,:])
		return Z

	
	def R(s):
		R = compute_features(G,Z(s))-b
		return R

	def J_R(s,B): 
		Z_new = np.copy(Z(s))
		J_R = np.empty((M,M))
		for k in np.arange(M):
			for j in np.arange(M):
				x = 0 
				for i in np.arange(N_B):
					x+=np.dot(A[N_B*B+i,j,:],evaluate_functions(J_G[k,:],Z_new[N_B*B+i,:]))
				J_R[k,j]=x/N_B
		return J_R


	def F(x):
		return np.sum(R(x)**2)

	def grad_F_and_HF(x,b):
		J_R_new = J_R(x,b)
		grad_F = np.dot(R(x),J_R_new)
		HF = np.dot(np.transpose(J_R_new), J_R_new)
		return (grad_F,HF)

	F_old = F(x)
	print("F(0) = " + str(F(np.zeros(G.shape[0]))))
	while done == False:
		bnit +=1
		q = np.random.permutation(B)
		for i in np.arange(B):
			nit += 1
			decreasing = False
			(grad_F,HF) = grad_F_and_HF(x,q[i])
			x_new = x - delta*np.linalg.solve((I+delta * HF), grad_F)
			F_new = F(x_new)
			decreasing = F_new <= F(x)
			if decreasing == False:
				print("Not decreasing...")
			x = np.copy(x_new)
			done = (F_new <= epsilon) or (nit >= 200)
			if done == True:
				break
			print("Solver iteration " + str(nit) + " completed, F(x) = " + str(F_new))
		print("Epoch " + str(bnit) + " completed.")

	print("number of iterations: " + str(nit))
	print("delta = " + str(delta))
	print("F(x) = " + str(F(x)))
	print("x = " + str(x))

	z = np.copy(x)

	def optimal_map(y):
		return y + np.dot(z,evaluate_functions(J_G,y))

	X_image = np.empty(X.shape)
	for i in np.arange(N_x):
		X_image[i,:]=optimal_map(X[i,:])

	return (x, optimal_map, X_image)




















