import matplotlib.pyplot as plt 
import numpy as np
from SOT import sample_OT_solver
from Stoch_LOT import local_OT_solver

mu1 = [0,0]
cov1 = [[1,0],[0,1]]

mu2 = [20,20]
cov2 = [[4,3],[3,4]]

mu3 = [0,20]
cov3 = [[4,-3],[-3,4]]

mu4 = [10,10]
cov4 = [[1,0], [0,1]]



np.random.seed(42)
x = np.random.multivariate_normal(mu1, cov1, 2000)
u = np.random.multivariate_normal(mu4,cov4,2000)
y = np.random.multivariate_normal(mu2,cov2, 2000)
z = np.random.multivariate_normal(mu3,cov3,2000)

print(x)
print(y)

plt.figure(1)
plt.plot(x[:,0], x[:,1], '.r')
plt.plot(y[:,0], y[:,1], '.b')
plt.plot(z[:,0], z[:,1], '.g')
plt.plot(u[:,0], u[:,1], '.y')
plt.show()


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

(G, J_G) = first_and_second_moments()

X = np.concatenate((x,u), axis=0)
Y = np.concatenate((y,z), axis=0)
s0 = np.zeros(G.shape[0])


(a, optimal_map, c) = local_OT_solver(G,J_G,X,Y, 10**-2, s0, 5)



plt.figure(2)
plt.plot(x[:,0], x[:,1], '.r', alpha=0.2)
plt.plot(c[:,0], c[:,1], '.k')
plt.plot(z[:,0], z[:,1], '.g', alpha=0.2)
plt.plot(y[:,0], y[:,1], '.b', alpha=0.2)
plt.plot(u[:,0], u[:,1], '.y', alpha=0.2)
plt.show()