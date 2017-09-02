import numpy as np 

def distance_matrix(S):
	D = np.empty((S.shape[0],S.shape[0]))
	for i in np.arange(S.shape[0]):
		for j in np.arange(S.shape[0]):
			D[i,j]=np.linalg.norm((S[i,:]-S[j,:]), ord=2, axis=0)

	return D