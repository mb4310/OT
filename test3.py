import numpy as np 
from numpy.linalg import inv
from numpy.linalg import norm
from scipy import misc

image_1 = 'test_OT_image1.tiff'
image_2 = 'test_OT_image2.tiff'

I_1 = misc.imread(image_1)
I_2 = misc.imread(image_2)

def re_index_image(Image):
	x_pixels,y_pixels = Image.shape
	A = np.empty([x_pixels * y_pixels, 3])
	for i in np.arange(x_pixels * y_pixels):
		x_index = i // y_pixels
		y_index = i % y_pixels
		A[i,0] = x_index / x_pixels
		A[i,1] = y_index / y_pixels
		A[i,2] = Image[x_index,y_index]
	return A

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

def compute_features(G, Y):  
	
	M = G.shape[0]
	N_y = Y.shape[0]
	b = np.empty((M))
	for i in np.arange(M):
		
		S = np.empty((N_y))

		for j in np.arange(N_y):
			S[j] = G[i](Y[j,:])
			
		b[i] = np.sum(S)/(N_y)

	return b

def compute_Ai(J_G,X):
	N_x = X.shape[0]
	A = np.empty((N_x, J_G.shape[0], J_G.shape[1]))
	for i in np.arange(N_x):
		A_i = evaluate_functions(J_G, X[i,:])
		A[i,:,:] = A_i 
	return A


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








G = np.array([g1, g2, g3, g4, g5, g6])

J_G = np.array([[f11,f12,f13], [f21, f22, f23], [f31, f32, f33], [f41, f42, f43], [f51, f52, f53], [f61, f62, f63]])

X = re_index_image(I_1)[::500,:]

Y = re_index_image(I_2)[::500,:]

b = compute_features(G,Y)

a = compute_features(G,X)

A = compute_Ai(J_G, X)

M = G.shape[0]

N_x = X.shape[0]

p = np.array([0, 0.001, 0.001, 0.001, 0.001, 0.001])

def Z(s): #accepts a 1-D array s of M dimensions, transforms (x_i)_[i=1]^{N_x} and returns (z_i)_{i=1}^{N_x} where z_i = x_i + s * A_i 
		Z = np.empty((X.shape[0],X.shape[1]))
		for i in np.arange(N_x):
			Z[i,:] = X[i,:] + np.dot(s,A[i,:,:])
		return Z

def R(s):
	R = compute_features(G,Z(s))-b
	return R

def J_R(s):
	Z_new = np.copy(Z(s))
	J_R = np.empty((M,M))
	for k in np.arange(M):
		for j in np.arange(M):
			x = 0 
			for i in np.arange(N_x):
				x+=np.dot(A[i,j,:],evaluate_functions(J_G[k,:],Z_new[i,:]))
			J_R[i,j]=x/N_x
	return J_R


def F(x):
	return np.sum(R(x)**2)

d = J_R.shape[1]
M = R.shape[0]
I = np.identity(d)
x = np.copy(p)
F_old = F(x)
nit = 0
delta = 0.2
done = False
epsilon = 10**(-9)

def HF_and_grad_F(x):
	J_R_new = J_R(x)
	grad_F = np.dot(R(x),J_R_new)
	HF = np.empty((d,d))
	for i in np.arange(d):
		for j in np.arange(d):
			HF[i,j]=np.dot(J_R_new[:,i],J_R_new[:,j])
	return (grad_F,HF)




while done == False:
		nit += 1
		decreasing = False
		while decreasing == False: 
			delta = delta/2
			x_new = x - delta*np.dot(inv((I + delta * evaluate_functions(HF,x))),evaluate_functions(grad_F,x))
			F_new = F(x_new)
			decreasing = F(x_new) <= F(x)
		x = np.copy(x_new)
		done = (np.abs(F(x)) <= epsilon) or (nit >= 200)
		delta = 1.9 * 2**nit * delta 


print("number of iterations: " + str(nit))
print("delta = " + str(delta))
print("F(x) = " + str(F(x)))
print("x = " + str(x))












