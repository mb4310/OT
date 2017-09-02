import numpy as np
from numpy.linalg import inv
from scipy import misc


d = 3

image_1 = 'checkerboard.jpg'
image_2 = 'grey_stripes.png'

I_1 = misc.imresize(misc.imread(image_1), (250,250))
I_2 = misc.imresize(misc.imread(image_2), (250,250))

I_1_new = np.empty((I_1.shape[0],I_1.shape[1]))
for i in np.arange(I_1.shape[0]):
	for j in np.arange(I_1.shape[1]):
		I_1_new[i,j] = I_1[i,j,0]

I_2_new = np.empty((I_2.shape[0],I_2.shape[1]))
for i in np.arange(I_2.shape[0]):
	for j in np.arange(I_2.shape[1]):
		I_2_new[i,j] = I_2[i,j,0]


print(I_1_new.shape)
print(I_2_new.shape)

def re_index_image(Image):
	x_pixels,y_pixels = Image.shape
	A = np.empty([x_pixels * y_pixels, 3])
	for i in np.arange(x_pixels * y_pixels):
		x_index = i // y_pixels
		y_index = i % y_pixels
		A[i,0] = x_index / x_pixels
		A[i,1] = y_index / y_pixels
		A[i,2] = Image[x_index,y_index] / np.amax(Image)
	return A

X = re_index_image(I_1_new)
Y = re_index_image(I_2_new)


def reconstruct_image(X, x_pixels,y_pixels, max_intensity):
	image = np.empty((x_pixels,y_pixels))
	N_x = X.shape[0]
	for i in np.arange(N_x):
		k = i // y_pixels
		j = i % y_pixels
		image[k,j] = X[i,2]*max_intensity
	return image

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

def grid_generator(a,b,c):   
	grid_centers = []
	bandwidths = []
	for i in np.arange(1, a):
		x = i/a
		for j in np.arange(1,b):
			y = j/b
			for k in np.arange(1, c):
				z = k  / c
				grid_centers.append([x,y,z])


	return grid_centers

def gaussian_pdf(x, mu, sigma, d):      ###CHECK THIS WORKS LOL
	return 1/(np.sqrt((2 * np.pi)**d)) * np.exp(-1 * np.linalg.norm(x-mu)**2/(2 * sigma **2))



grid = np.array(grid_generator(3,3,3))

bandwidths = []

for k in np.arange(grid.shape[0]):
	bandwidths.append(1.0)


def kernels_definer(i):
	def ith_kernel(x):
		return gaussian_pdf(x, grid[i,:], bandwidths[i], 3)
	return ith_kernel

K = np.empty(grid.shape[0], dtype = object)

for k in np.arange(grid.shape[0]):
	K[k] = kernels_definer(k)

J_K = np.empty((grid.shape[0], 3), dtype = object)

def kernels_grad_definer(i,j):
	def g_ij(x):
		return  -1 * (x[j]-grid[i][j])/(bandwidths[i]**2) * gaussian_pdf(x, grid[i,:], bandwidths[i], 3)
	return g_ij


for i in np.arange(grid.shape[0]):
	for j in np.arange(d):
		J_K[i,j] = kernels_grad_definer(i,j)



(M,J_M) = first_and_second_moments()




G = np.concatenate((M,K))

J_G = np.concatenate((J_M,J_K), axis=0)

X = re_index_image(I_1_new)

Y = re_index_image(I_2)

b = compute_features(G,Y)

a = compute_features(G,X)

print("b = " + str(b))
print("a = " + str(a))

A = compute_Ai(J_G, X)
print("A has been computed...")

M = G.shape[0]

N_x = X.shape[0]
print("N_x = " + str(N_x))

p = np.zeros(G.shape[0])


def Z(s):  
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
			J_R[k,j]=x/N_x
	return J_R


def F(x):
	return np.sum(R(x)**2)

d = X.shape[1]
M = G.shape[0]
I = np.identity(M)
x = np.copy(p)
F_old = F(x)
nit = 0
delta = 0.2
done = False
epsilon = 10**(-8)

def grad_F_and_HF(x):
	J_R_new = J_R(x)
	grad_F = np.dot(R(x),J_R_new)
	HF = np.dot(np.transpose(J_R_new), J_R_new)
	return (grad_F,HF)

print("F(0) is: " + str(F(p)))

while done == False:
		nit += 1
		decreasing = False
		while decreasing == False:
			(grad_F,HF) = grad_F_and_HF(x)
			delta = delta/2
			x_new = x - delta*np.linalg.solve((I+delta * HF), grad_F)
			F_new = F(x_new)
			decreasing = F_new <= F(x)
			if decreasing == False:
				print("Not decreasing...")
		x = np.copy(x_new)
		done = (F_new <= epsilon) or (nit >= 200)
		delta = 1.9 * 2**nit * delta 
		print("iteration " + str(nit) + " completed, F(x) = " + str(F_new))

print("Number of iterations: " + str(nit))
print("delta = " + str(delta))
print("F(x) = " + str(F_new))
print("x = " + str(x))

z = np.copy(x)

def optimal_map(y):
	return y + np.dot(z,evaluate_functions(J_G,y))

X_transform = np.empty(X.shape)


for i in np.arange(N_x):
	X_transform[i,:] = optimal_map(X[i,:])



I_transformed = reconstruct_image(X_transform, I_1_new.shape[0], I_1_new.shape[1], np.amax(I_1_new))


misc.imsave('new_image.png', I_transformed)






