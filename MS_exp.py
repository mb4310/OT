import matplotlib.pyplot as plt
import numpy as np 



np.random.seed(42)

mean_i1 = [0,0]
cov_i1 = [[1,0],[0,1]]

mean_i2= [0,5]
cov_i2 = [[1,0],[0,1]]

mean_i3 = [0,10]
cov_i3 = [[1,0],[0,1]]

mean_i4 = [10,0]
cov_i4 = [[1,0],[0,1]]

mean_i5 = [10,5]
cov_i5 = [[1,0],[0,1]]

mean_i6 = [10,10]
cov_i6 = [[1,0],[0,1]]

x1 = np.random.multivariate_normal(mean_i1, cov_i1, 1000)
x2 = np.random.multivariate_normal(mean_i2, cov_i2, 1000)
x3 = np.random.multivariate_normal(mean_i3, cov_i3, 1000)
x4 = np.random.multivariate_normal(mean_i4, cov_i4, 1000)
x5 = np.random.multivariate_normal(mean_i5, cov_i5, 1000)
x6 = np.random.multivariate_normal(mean_i6, cov_i6, 1000)

X = np.concatenate((x1,x2,x3,x4,x5,x6), axis=0)



plt.figure()
plt.scatter(X[:,0], X[:,1], alpha=0.2, marker='.')
plt.show()