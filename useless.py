import matplotlib.pyplot as plt 
import numpy as np

mu1 = [0,0]
cov1 = [[3,0],[0,3]]

mu2 = [0,15]
cov2 = [[3,0],[0,3]]
np.random.seed(42)

x = np.random.multivariate_normal(mu1, cov1, 5000)
y = np.random.multivariate_normal(mu2, cov2, 5000)

def grid_generator(a,b):   
	grid_centers = []
	bandwidths = []
	for i in np.arange(1, a):
		x = i/a
		for j in np.arange(1,b):
			y = j/b
			grid_centers.append([x,y])

	return np.array(grid_centers)

grid  = grid_generator(4,4)


c1 = [0.1,0.1]
c2 = [2.7,0]
c3 = [0,5]


fig = plt.figure()

plt.subplot(211)
plt.title('Case 1: Bandwidth too small')
plt.scatter(x[:,0], x[:,1], alpha=0.2, marker='.')
plt.scatter(y[:,0], y[:,1], alpha=0.2, marker='.')
plt.plot(3.1, 0.1, 'xk', label='Case 2')
plt.scatter(grid[:,0],grid[:,1], marker='x')

plt.subplot(212)
plt.title('Case 2: Bandwidth too large')
plt.scatter(x[:,0], x[:,1], alpha=0.2, marker='.')
plt.scatter(y[:,0], y[:,1], alpha=0.2, marker='.')
plt.plot(0.1, 7.5, 'xk', label='Case 3')
plt.scatter(grid[:,0],grid[:,1], marker='x')

plt.show()
