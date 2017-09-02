import numpy as np
from numpy.linalg import inv
from scipy import misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

d = 3
h = 5



image_1 = 'checkerboard.jpg'
image_2 = 'grey_stripes.jpg'

I_1 = misc.imresize(misc.imread(image_1), (100,100))
I_2 = misc.imresize(misc.imread(image_2), (100,100))

I_1_new = np.empty((I_1.shape[0],I_1.shape[1]))
for i in np.arange(I_1.shape[0]):
	for j in np.arange(I_1.shape[1]):
		I_1_new[i,j] = I_1[i,j,0]

I_2_new = np.empty((I_2.shape[0],I_2.shape[1]))
for i in np.arange(I_2.shape[0]):
	for j in np.arange(I_2.shape[1]):
		I_2_new[i,j] = I_2[i,j,0]

def re_index_image(Image):
	x_pixels,y_pixels = Image.shape
	A = np.empty([x_pixels * y_pixels, 3])
	for i in np.arange(x_pixels * y_pixels):
		x_index = i // y_pixels
		y_index = i % y_pixels
		A[i,0] = x_index 
		A[i,1] = y_index 
		A[i,2] = Image[x_index,y_index] 
	return A

X = re_index_image(I_1_new)
Y = re_index_image(I_2_new)

def T(x):      #Mean zero Gaussian kernel
	return np.exp(-1 * np.linalg.norm(x)**2/(2 * h**2))


def kernel_mean(s,X,T): #X is the set of data, s is the sample mean. This implementation is atrocious(!)
	N_x = X.shape[0]
	x = 0
	y = 0
	L = s - X
	for i in np.arange(N_x):
		x+=T(L[i,:])*X[i,:]
	for i in np.arange(N_x):
		y+=T(L[i,:])
	return x/y

def mean_shift_cluster(G, X, K, epsilon):				#G is an array of your initial cluster centers, X is data, epsilon is stopping threshold, K is desired kernel
	G_new = np.copy(G)
	done = False
	nit = 0
	N_x = X.shape[0]
	N_G = G.shape[0]
	while done == False:
		nit +=1 
		new_centers = np.empty(G.shape)
		for i in np.arange(N_G):
			new_centers[i,:] = kernel_mean(G_new[i,:], X, K)
		change = np.sum(np.linalg.norm((new_centers - G_new), ord=2, axis=1))
		G_new = np.copy(new_centers)
		done = (change <= epsilon) or (nit >= 25)
		print("iteration " + str(nit) + " completed, " + "epsilon = " + str(change))
	return G_new

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



grid1 = np.array(grid_generator(4,4,4))
grid2 = np.array(grid_generator(4,4,4))


S1 = mean_shift_cluster(grid1, X, T, 10**-1)
S2 = mean_shift_cluster(grid2, Y, T, 10**-1)
print(S1)
print(S2)
print("total change in first grid was " + str(np.sum(np.linalg.norm((grid1-S1), ord=2, axis=1))))
print("total change in second grid was " + str(np.sum(np.linalg.norm((grid2-S2), ord=2, axis=1))))


fig1 = plt.figure(1)
fig2 = plt.figure(2)
ax1 = fig1.add_subplot(111, projection='3d')
ax2 = fig2.add_subplot(111, projection='3d')

ax1.scatter(S1[:,0],S1[:,1],S1[:,2], c='k')
ax2.scatter(S2[:,0],S2[:,1],S2[:,2], c='r')
ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z Label')
ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')
ax2.set_zlabel('Z Label')
plt.show()







