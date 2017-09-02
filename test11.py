import numpy as np


S = np.array([[1,0,0],[2,0,1],[3,0,0],[4,0,1]])

def distance_matrix(S):
	if S.ndim == 2:
		D = np.empty((S.shape[0],S.shape[0]))
		for i in np.arange(S.shape[0]):
			for j in np.arange(S.shape[0]):
				D[i,j]=np.linalg.norm((S[i,:]-S[j,:]), ord=2, axis=0)

	else: 
		D = np.empty((S.shape[0],S.shape[0]))
		for i in np.arange(S.shape[0]):
			for j in np.arange(S.shape[0]):
				D[i,j]=np.linalg.norm((S[i]-S[j]))

	return D

print(S)

S_new = np.delete()

